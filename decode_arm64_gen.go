//go:build ignore

// Command gen produces decode_arm64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) decoder for arm64 using NEON, 4 groups (20 chars -> 16
// bytes) per iteration.
//
// Each word lane l decodes one 5-char group with a base-85 multiply-accumulate
// (((c0*85+c1)*85+c2)*85+c3)*85+c4 (VMUL multiply-low word + VADD), the inverse
// of the encoder's reciprocal-divide; the low 32 bits are exact for any valid
// encoding. The integer vector multiply (VMUL) is the same mnemonic Go's arm64
// assembler only gained in Go 1.27, so this kernel is guarded //go:build go1.27
// and the stable build falls back to the scalar decode_generic.go path on arm64.
//
// Gather: the five chars of lane l sit at byte addresses 5l+k (k=0..4). For each
// k, a VLD1 of src+k loads bytes k..k+15 and a single shared VTBL control
// {0,_,_,_,5,_,_,_,10,_,_,_,15,_,_,_} places window byte 5l into the LOW byte of
// word lane l (byte address 4l, little-endian); gaps use an index with bit7 set
// (>=0x80), which VTBL writes as 0. So lane l's word equals the char; '!' (33)
// is subtracted with VSUB.
//
// Output: after the MAC, word lane l holds v with byte at address 4l = v (LSB);
// a per-lane byte-reverse VREV32 lays it out big-endian, then one VST1 writes
// the 16 bytes.
//
// The caller hands the kernel only clean runs of whole 5-char groups of regular
// base-85 digits ('!'..'u'); whitespace, "z", the short/invalid trailing group,
// flush, and CorruptInputError offsets are handled by the caller via the stdlib.
//
// Run: GOWORK=off go run decode_arm64_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func le4(v uint32) []byte {
	b := make([]byte, 16)
	for i := 0; i < 4; i++ {
		b[i*4+0] = byte(v)
		b[i*4+1] = byte(v >> 8)
		b[i*4+2] = byte(v >> 16)
		b[i*4+3] = byte(v >> 24)
	}
	return b
}

func main() {
	f := emit.NewFile("arm64")

	c33 := f.Data("d85c33", le4(33))
	c85 := f.Data("d85c85", le4(85))
	// Gather control: window byte 5l -> word lane l low byte (addr 4l). Gaps use
	// bit7 set (-> zero from VTBL).
	const z = 0x80
	gath := f.Data("d85gath", []byte{0, z, z, z, 5, z, z, z, 10, z, z, z, 15, z, z, z})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := arm64.NewFunc("decodeGroups", sig, 0)
	b.LoadArg("dst_base", "R0").LoadArg("src_base", "R1").LoadArg("blocks", "R2").
		Raw("MOVD $%s(SB), R3", c33).Raw("VLD1 (R3), [V22.B16]").
		Raw("MOVD $%s(SB), R3", c85).Raw("VLD1 (R3), [V21.B16]").
		Raw("MOVD $%s(SB), R3", gath).Raw("VLD1 (R3), [V23.B16]").
		Raw("CBZ R2, done").
		Label("loop").
		// Gather C0..C4 from windows src+k via VTBL, subtract '!'.
		Raw("VLD1 (R1), [V0.B16]").Raw("VTBL V23.B16, [V0.B16], V1.B16").Raw("VSUB V22.S4, V1.S4, V1.S4").                       // C0
		Raw("ADD $1, R1, R3").Raw("VLD1 (R3), [V0.B16]").Raw("VTBL V23.B16, [V0.B16], V2.B16").Raw("VSUB V22.S4, V2.S4, V2.S4"). // C1
		Raw("ADD $2, R1, R3").Raw("VLD1 (R3), [V0.B16]").Raw("VTBL V23.B16, [V0.B16], V3.B16").Raw("VSUB V22.S4, V3.S4, V3.S4"). // C2
		Raw("ADD $3, R1, R3").Raw("VLD1 (R3), [V0.B16]").Raw("VTBL V23.B16, [V0.B16], V4.B16").Raw("VSUB V22.S4, V4.S4, V4.S4"). // C3
		Raw("ADD $4, R1, R3").Raw("VLD1 (R3), [V0.B16]").Raw("VTBL V23.B16, [V0.B16], V5.B16").Raw("VSUB V22.S4, V5.S4, V5.S4"). // C4
		// MAC: v = ((((C0*85+C1)*85+C2)*85+C3)*85+C4.
		Raw("VMUL V21.S4, V1.S4, V1.S4").Raw("VADD V2.S4, V1.S4, V1.S4").
		Raw("VMUL V21.S4, V1.S4, V1.S4").Raw("VADD V3.S4, V1.S4, V1.S4").
		Raw("VMUL V21.S4, V1.S4, V1.S4").Raw("VADD V4.S4, V1.S4, V1.S4").
		Raw("VMUL V21.S4, V1.S4, V1.S4").Raw("VADD V5.S4, V1.S4, V1.S4").
		// Big-endian byte reverse + store.
		Raw("VREV32 V1.B16, V0.B16").
		Raw("VST1 [V0.B16], (R0)").
		Raw("ADD $20, R1, R1").Raw("ADD $16, R0, R0").Raw("SUB $1, R2, R2").
		Raw("CBNZ R2, loop").
		Label("done").Ret()
	f.Add(b.Func())

	out := strings.Replace(f.String(), "//go:build arm64\n", "//go:build arm64 && go1.27\n", 1)
	if err := os.WriteFile("decode_arm64.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_arm64.s")
}
