//go:build ignore

// Command gen produces decode_ppc64x.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) decoder for ppc64le using the VSX/AltiVec vector facility,
// 4 groups (20 chars -> 16 bytes) per iteration. It uses ISA-3.0 (POWER9)
// instructions (LXVB16X/STXVB16X), which raise SIGILL on POWER8, so the
// dispatcher gates it on cpu.PPC64.IsPOWER9 and falls back to a scalar loop
// (decodeScalar) on POWER8.
//
// Each word lane l decodes one 5-char group with a base-85 multiply-accumulate
// (((c0*85+c1)*85+c2)*85+c3)*85+c4 (VMULUWM multiply-low + VADDUWM), the inverse
// of the encoder's reciprocal-divide; the low 32 bits are exact for any valid
// encoding.
//
// Gather: the five chars of lane l sit at byte addresses 5l+k (k=0..4). For each
// k, an LXVB16X of src+k loads bytes k..k+15 (byte-order-correct: its in-register
// words are big-endian), and a single shared VPERM control places window byte 5l
// into the LOW byte of word lane l's big-endian value (byte address 4l+3),
// zeroing the rest via gap index 16 -> the all-zero VPERM operand (ppc VPERM has
// no PSHUFB-style 0x80 zeroing). So lane l's word equals the char; '!' (33) is
// subtracted with VSUBUWM.
//
// Output: after the MAC, word lane l holds the decoded value v; STXVB16X writes
// the in-register big-endian bytes, i.e. byte address 4l = v>>24, so the 16 bytes
// come out big-endian with no reversal.
//
// VSX<->VMX aliasing: vectors load with LXVB16X into VS(32+k), operated as Vk;
// the store is STXVB16X of VS(32+k).
//
// The caller hands the kernel only clean runs of whole 5-char groups of regular
// base-85 digits ('!'..'u'); whitespace, "z", the short/invalid trailing group,
// flush, and CorruptInputError offsets are handled by the caller via the stdlib.
//
// Run: GOWORK=off go run decode_ppc64x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
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
	f := emit.NewFile("ppc64le")

	c33 := f.Data("d85c33", be4(33))
	c85 := f.Data("d85c85", be4(85))
	// Gather control (byte-address indexed). Window byte 5l -> word lane l low byte
	// (address 4l+3); gaps use index 16 -> the all-zero second VPERM operand.
	const z = 16
	gath := f.Data("d85gath", []byte{z, z, z, 0, z, z, z, 5, z, z, z, 10, z, z, z, 15})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := ppc64.NewFunc("decodeGroups", sig, 0)
	b.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("blocks", "R5").
		Raw("MOVD $%s(SB), R6", c33).Raw("LXVB16X (R6)(R0), VS52").  // V20 = 33
		Raw("MOVD $%s(SB), R6", c85).Raw("LXVB16X (R6)(R0), VS53").  // V21 = 85
		Raw("MOVD $%s(SB), R6", gath).Raw("LXVB16X (R6)(R0), VS54"). // V22 = gather ctrl
		Raw("VSPLTISW $0, V19").                                     // zero (VPERM gap source)
		Raw("CMP R5, $0").Raw("BEQ done").
		Label("loop").
		// Gather C0..C4 from windows src+k via VPERM, subtract '!'.
		Raw("ADD $0, R4, R6").Raw("LXVB16X (R6)(R0), VS32").Raw("VPERM V0, V19, V22, V0").Raw("VSUBUWM V0, V20, V0"). // C0
		Raw("ADD $1, R4, R6").Raw("LXVB16X (R6)(R0), VS33").Raw("VPERM V1, V19, V22, V1").Raw("VSUBUWM V1, V20, V1"). // C1
		Raw("ADD $2, R4, R6").Raw("LXVB16X (R6)(R0), VS34").Raw("VPERM V2, V19, V22, V2").Raw("VSUBUWM V2, V20, V2"). // C2
		Raw("ADD $3, R4, R6").Raw("LXVB16X (R6)(R0), VS35").Raw("VPERM V3, V19, V22, V3").Raw("VSUBUWM V3, V20, V3"). // C3
		Raw("ADD $4, R4, R6").Raw("LXVB16X (R6)(R0), VS36").Raw("VPERM V4, V19, V22, V4").Raw("VSUBUWM V4, V20, V4"). // C4
		// MAC: v = ((((C0*85+C1)*85+C2)*85+C3)*85+C4.
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V1, V0").
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V2, V0").
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V3, V0").
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V4, V0").
		// Big-endian store: lane l word -> output bytes 4l..4l+3, no reversal.
		Raw("STXVB16X VS32, (R3)(R0)").
		Raw("ADD $20, R4").Raw("ADD $16, R3").Raw("ADD $-1, R5").
		Raw("CMP R5, $0").Raw("BNE loop").
		Label("done").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("decode_ppc64x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_ppc64x.s")
}
