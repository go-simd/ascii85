//go:build ignore

// Command gen produces decode_s390x.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) decoder for s390x using the z/Architecture vector
// facility, 4 groups (20 chars -> 16 bytes) per iteration.
//
// Each fullword lane l decodes one 5-char group with a base-85
// multiply-accumulate (((c0*85+c1)*85+c2)*85+c3)*85+c4 (VMLF multiply-low +
// VAF), the inverse of the encoder's reciprocal-divide. The low 32 bits are
// exact for any valid encoding.
//
// Gather: the five chars of lane l sit at byte addresses 5l+k (k=0..4). For each
// k, a VL of src+k brings bytes k..k+15 into a register, and a single shared
// VPERM control places window byte 5l into the LOW byte of fullword lane l (byte
// address 4l+3 on big-endian s390x), zeroing the rest via gap index 16 -> the
// all-zero second VPERM operand. So lane l's fullword equals the char (0..127);
// '!' (33) is subtracted with VSF.
//
// s390x is BIG-ENDIAN, which makes the OUTPUT trivial: after the MAC, fullword
// lane l holds the decoded value v with byte address 4l = v>>24, so a single VST
// writes the 16 output bytes in big-endian order with no byte reversal (verified
// position-dependently under qemu: lane l -> output bytes 4l..4l+3).
//
// The caller hands the kernel only clean runs of whole 5-char groups of regular
// base-85 digits ('!'..'u'); whitespace, "z", the short/invalid trailing group,
// flush, and CorruptInputError offsets are handled by the caller via the stdlib.
//
// Run: GOWORK=off go run decode_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func be4(v uint32) []byte {
	b := make([]byte, 16)
	for i := 0; i < 4; i++ {
		b[i*4+0] = byte(v >> 24)
		b[i*4+1] = byte(v >> 16)
		b[i*4+2] = byte(v >> 8)
		b[i*4+3] = byte(v)
	}
	return b
}

func main() {
	f := emit.NewFile("s390x")

	c33 := f.Data("d85c33", be4(33))
	c85 := f.Data("d85c85", be4(85))
	// Gather control (byte-address indexed). Window byte 5l -> fullword lane l low
	// byte (address 4l+3, big-endian); gaps use index 16 -> zero second operand.
	const z = 16
	gath := f.Data("d85gath", []byte{z, z, z, 0, z, z, z, 5, z, z, z, 10, z, z, z, 15})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	g := s390x.NewFunc("decodeGroups", sig, 0)
	g.LoadArg("dst_base", "R1").LoadArg("src_base", "R2").LoadArg("blocks", "R3").
		Raw("MOVD $%s(SB), R4", c33).Raw("VL (R4), V20").
		Raw("MOVD $%s(SB), R4", c85).Raw("VL (R4), V21").
		Raw("MOVD $%s(SB), R4", gath).Raw("VL (R4), V22").
		Raw("VZERO V19").
		Raw("CMPBEQ R3, $0, done").
		Label("loop").
		// Gather C0..C4 from windows src+k via VPERM, subtract '!'.
		Raw("VL 0(R2), V0").Raw("VPERM V0, V19, V22, V0").Raw("VSF V20, V0, V0"). // C0
		Raw("VL 1(R2), V1").Raw("VPERM V1, V19, V22, V1").Raw("VSF V20, V1, V1"). // C1
		Raw("VL 2(R2), V2").Raw("VPERM V2, V19, V22, V2").Raw("VSF V20, V2, V2"). // C2
		Raw("VL 3(R2), V3").Raw("VPERM V3, V19, V22, V3").Raw("VSF V20, V3, V3"). // C3
		Raw("VL 4(R2), V4").Raw("VPERM V4, V19, V22, V4").Raw("VSF V20, V4, V4"). // C4
		// MAC: v = ((((C0*85+C1)*85+C2)*85+C3)*85+C4.
		Raw("VMLF V0, V21, V0").Raw("VAF V1, V0, V0").
		Raw("VMLF V0, V21, V0").Raw("VAF V2, V0, V0").
		Raw("VMLF V0, V21, V0").Raw("VAF V3, V0, V0").
		Raw("VMLF V0, V21, V0").Raw("VAF V4, V0, V0").
		// Big-endian store: lane l fullword -> output bytes 4l..4l+3, no reversal.
		Raw("VST V0, (R1)").
		Raw("ADD $20, R2").Raw("ADD $16, R1").Raw("ADD $-1, R3").
		Raw("CMPBNE R3, $0, loop").
		Label("done").Ret()
	f.Add(g.Func())

	if err := os.WriteFile("decode_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_s390x.s")
}
