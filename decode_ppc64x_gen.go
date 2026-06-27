//go:build ignore

// Command gen produces decode_ppc64x.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) decoder for ppc64le using the VSX/AltiVec vector facility,
// 4 groups (20 chars -> 16 bytes) per iteration. It is built from ISA-2.07
// (POWER8-baseline) ops only, so it runs natively on POWER8 (no POWER9 gate).
// The byte-order-correct vector loads/stores a codec wants would be the ISA-3.0
// LXVB16X/STXVB16X; those are emitted instead as an ISA-2.07 LXVD2X/STXVD2X plus
// one VPERM against a fixed byte-reversal control vrev (see emitLoadB16/
// emitStoreB16 below). vrev is bootstrapped by a plain LXVD2X of {0..15}, whose
// scrambled load layout is exactly the involution VPERM needs to reproduce the
// LXVB16X element order. Verified on cfarm433 (POWER9) and cfarm112 (POWER8E).
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

// vrev byte-reversal control lives in V31 (VS63) for the whole kernel.
const vrevVS = "VS63"
const vrevV = "V31"

// emitLoadB16 emits the ISA-2.07 equivalent of "LXVB16X (addrExpr), vs".
func emitLoadB16(bld *ppc64.Builder, addrExpr, vs, v string) {
	bld.Raw("LXVD2X %s, %s", addrExpr, vs).
		Raw("VPERM %s, %s, %s, %s", v, v, vrevV, v)
}

// emitStoreB16 emits the ISA-2.07 equivalent of "STXVB16X vs, (addrExpr)" using
// the V30/VS62 scratch; v is left clobbered.
func emitStoreB16(bld *ppc64.Builder, vs, v, addrExpr string) {
	bld.Raw("VPERM %s, %s, %s, V30", v, v, vrevV).
		Raw("STXVD2X VS62, %s", addrExpr)
}

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
	// Byte-reversal control for the LXVB16X/STXVB16X emulation (loaded via plain
	// LXVD2X into the involution layout VPERM needs).
	rev := f.Data("d85rev", []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := ppc64.NewFunc("decodeGroups", sig, 0)
	bld := b.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("blocks", "R5").
		Raw("MOVD $%s(SB), R6", rev).Raw("LXVD2X (R6)(R0), %s", vrevVS) // V31 = vrev
	bld.Raw("MOVD $%s(SB), R6", c33)
	emitLoadB16(bld, "(R6)(R0)", "VS52", "V20") // V20 = 33
	bld.Raw("MOVD $%s(SB), R6", c85)
	emitLoadB16(bld, "(R6)(R0)", "VS53", "V21") // V21 = 85
	bld.Raw("MOVD $%s(SB), R6", gath)
	emitLoadB16(bld, "(R6)(R0)", "VS54", "V22") // V22 = gather ctrl
	bld.Raw("VSPLTISW $0, V19").                // zero (VPERM gap source)
		Raw("CMP R5, $0").Raw("BEQ done").
		Label("loop")
	// Gather C0..C4 from windows src+k. Each load is the ISA-2.07 LXVB16X
	// emulation (LXVD2X + VPERM vrev -> identity layout in V0..V4), then the
	// existing gather VPERM places window byte 5l into word lane l, then -'!'.
	emitLoadB16(bld, "(R4)(R0)", "VS32", "V0")
	bld.Raw("VPERM V0, V19, V22, V0").Raw("VSUBUWM V0, V20, V0") // C0
	bld.Raw("ADD $1, R4, R6")
	emitLoadB16(bld, "(R6)(R0)", "VS33", "V1")
	bld.Raw("VPERM V1, V19, V22, V1").Raw("VSUBUWM V1, V20, V1") // C1
	bld.Raw("ADD $2, R4, R6")
	emitLoadB16(bld, "(R6)(R0)", "VS34", "V2")
	bld.Raw("VPERM V2, V19, V22, V2").Raw("VSUBUWM V2, V20, V2") // C2
	bld.Raw("ADD $3, R4, R6")
	emitLoadB16(bld, "(R6)(R0)", "VS35", "V3")
	bld.Raw("VPERM V3, V19, V22, V3").Raw("VSUBUWM V3, V20, V3") // C3
	bld.Raw("ADD $4, R4, R6")
	emitLoadB16(bld, "(R6)(R0)", "VS36", "V4")
	bld.Raw("VPERM V4, V19, V22, V4").Raw("VSUBUWM V4, V20, V4"). // C4
		// MAC: v = ((((C0*85+C1)*85+C2)*85+C3)*85+C4.
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V1, V0").
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V2, V0").
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V3, V0").
		Raw("VMULUWM V0, V21, V0").Raw("VADDUWM V0, V4, V0")
	// Big-endian store: lane l word -> output bytes 4l..4l+3.
	emitStoreB16(bld, "VS32", "V0", "(R3)(R0)")
	bld.Raw("ADD $20, R4").Raw("ADD $16, R3").Raw("ADD $-1, R5").
		Raw("CMP R5, $0").Raw("BNE loop").
		Label("done").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("decode_ppc64x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_ppc64x.s")
}
