//go:build ignore

// Command gen produces encode_arm64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) encoder for arm64 using NEON, 4 groups (16 input bytes ->
// 20 chars) per iteration.
//
// The base-85 challenge is the repeated division by 85. The kernel avoids it
// with a reciprocal multiply: for any 32-bit v, v/85 == (v*0xC0C0C0C1)>>32>>6
// (verified exact over all 2^32 values), remainder = v - q*85. Five such steps
// yield the five base-85 digits per 4-byte group. The 32-bit mulhi needed for
// the >>32 step is an INTEGER vector multiply: Go's arm64 assembler exposed only
// the polynomial VPMULL until the integer VMUL / VUMULL / VUMULL2 mnemonics were
// upstreamed in Go 1.27, so this kernel is guarded //go:build go1.27 and the
// stable build falls back to the scalar encode_generic.go path on arm64.
//
// mulhi: VUMULL widens 32x32->64 for lanes 0,1 (.S2/.D2) and VUMULL2 for lanes
// 2,3 (.S4/.D2); VUZP2 of the two 64-bit-product registers (read as .S4) picks
// the odd (high) 32-bit halves of all four products into one .S4 register, the
// per-lane high32 the reciprocal divide wants.
//
// arm64 is little-endian, so a VREV32 byte-reverse within each 32-bit lane turns
// the loaded word into the big-endian value v the encoder needs.
//
// Scatter: each digit byte sits in the low byte of its word lane (byte address
// 4l, little-endian). The five digit vectors are packed D0..D3 at stride 4 (byte
// shifts via VSHL) and the stride-4 word is expanded to the stride-5 output with
// VTBL controls into two overlapping 16-byte windows (bytes 0..15 at dst and
// 4..19 at dst+4); a zero source vector supplies the gaps, and the 5th digit
// (D4) is merged in with its own VTBL control. Two VST1 stores write the 20
// bytes.
//
// The all-zero "z" shortcut, the (n&3) leftover groups, and the trailing
// fragment are handled by the caller via the stdlib, so this kernel only ever
// sees non-zero groups and always writes exactly 5 chars per group.
//
// Run: GOWORK=off go run encode_arm64_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

// le4 stores a 32-bit constant little-endian into all four word lanes so the
// in-register word value (arm64 is LE) equals v.
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

	mrec := f.Data("a85mrec", le4(0xC0C0C0C1))
	c85 := f.Data("a85c85", le4(85))
	c33 := f.Data("a85c33", le4(33))
	// Scatter controls. Each digit byte sits in the low byte of its word lane,
	// i.e. byte address 4l (little-endian). Gaps use an index with bit7 set
	// (>=0x80), which VTBL writes as 0. Two overlapping windows: A = bytes 0..15
	// (at dst), B = bytes 4..19 (at dst+4). The packed digits are a single
	// 16-byte register (one VTBL source); D4 is a separate register.
	const z = 0x80
	expA := f.Data("a85expA", []byte{0, 1, 2, 3, z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12})
	d4A := f.Data("a85d4A", []byte{z, z, z, z, 0, z, z, z, z, 4, z, z, z, z, 8, z})
	expB := f.Data("a85expB", []byte{z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12, 13, 14, 15, z})
	d4B := f.Data("a85d4B", []byte{0, z, z, z, z, 4, z, z, z, z, 8, z, z, z, z, 12})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := arm64.NewFunc("encodeGroups", sig, 0)
	bld := b.LoadArg("dst_base", "R0").LoadArg("src_base", "R1").LoadArg("blocks", "R2").
		Raw("MOVD $%s(SB), R3", mrec).Raw("VLD1 (R3), [V20.B16]").
		Raw("MOVD $%s(SB), R3", c85).Raw("VLD1 (R3), [V21.B16]").
		Raw("MOVD $%s(SB), R3", c33).Raw("VLD1 (R3), [V22.B16]").
		Raw("MOVD $%s(SB), R3", expA).Raw("VLD1 (R3), [V24.B16]").
		Raw("MOVD $%s(SB), R3", d4A).Raw("VLD1 (R3), [V25.B16]").
		Raw("MOVD $%s(SB), R3", expB).Raw("VLD1 (R3), [V26.B16]").
		Raw("MOVD $%s(SB), R3", d4B).Raw("VLD1 (R3), [V27.B16]").
		Raw("VEOR V28.B16, V28.B16, V28.B16"). // zero (VTBL gap source)
		Raw("CBZ R2, done").
		Label("loop").
		Raw("VLD1 (R1), [V0.B16]").
		Raw("VREV32 V0.B16, V0.B16") // V0 = per-lane big-endian value v

	// digit: q = vReg/85 -> qReg ; remainder vReg - q*85 -> rReg. Scratch V16,V17.
	digit := func(vReg, qReg, rReg string) {
		bld.Raw("VUMULL V20.S2, %s.S2, V16.D2", vReg). // products of lanes 0,1
								Raw("VUMULL2 V20.S4, %s.S4, V17.D2", vReg). // products of lanes 2,3
								Raw("VUZP2 V17.S4, V16.S4, V16.S4").        // V16 = high32(v*mrec) per lane
								Raw("VUSHR $6, V16.S4, %s.S4", qReg).       // q = high32 >> 6
								Raw("VMUL V21.S4, %s.S4, V17.S4", qReg).    // V17 = q*85 (low word)
								Raw("VSUB V17.S4, %s.S4, %s.S4", vReg, rReg)
	}

	digit("V0", "V1", "V5") // q1=V1, r4=V5
	digit("V1", "V2", "V4") // q2=V2, r3=V4
	digit("V2", "V3", "V6") // q3=V3, r2=V6
	digit("V3", "V7", "V8") // q4=V7 (=r0), r1=V8
	// D0=V7, D1=V8, D2=V6, D3=V4, D4=V5 ; digit byte at low byte of word (addr 4l).
	bld.
		// add '!' (33) to every digit.
		Raw("VADD V22.S4, V7.S4, V7.S4").
		Raw("VADD V22.S4, V8.S4, V8.S4").
		Raw("VADD V22.S4, V6.S4, V6.S4").
		Raw("VADD V22.S4, V4.S4, V4.S4").
		Raw("VADD V22.S4, V5.S4, V5.S4").
		// pack4 (stride 4): byte 4l+j = Dj. On little-endian the digit is already
		// at byte 4l, so shift Dj LEFT (toward higher addresses) by j bytes =
		// VSHL per word by 8*j bits (stays within the 4-byte lane: j<=3).
		Raw("VSHL $8, V8.S4, V8.S4").  // D1 byte -> 4l+1
		Raw("VSHL $16, V6.S4, V6.S4"). // D2 byte -> 4l+2
		Raw("VSHL $24, V4.S4, V4.S4"). // D3 byte -> 4l+3
		Raw("VORR V8.B16, V7.B16, V7.B16").
		Raw("VORR V6.B16, V7.B16, V7.B16").
		Raw("VORR V4.B16, V7.B16, V7.B16"). // V7 = packed4 (D0..D3, stride 4)
		// expand stride-4 -> stride-5 into two overlapping windows; merge D4 (=V5).
		Raw("VTBL V24.B16, [V7.B16], V9.B16").  // window A: exp
		Raw("VTBL V25.B16, [V5.B16], V10.B16"). // window A: d4
		Raw("VORR V10.B16, V9.B16, V9.B16").    // V9 = output bytes 0..15
		Raw("VTBL V26.B16, [V7.B16], V11.B16"). // window B: exp
		Raw("VTBL V27.B16, [V5.B16], V12.B16"). // window B: d4
		Raw("VORR V12.B16, V11.B16, V11.B16").  // V11 = output bytes 4..19
		Raw("VST1 [V9.B16], (R0)").             // store bytes 0..15 at dst
		Raw("ADD $4, R0, R0").
		Raw("VST1 [V11.B16], (R0)"). // store bytes 4..19 at dst+4
		Raw("ADD $16, R0, R0").
		Raw("ADD $16, R1, R1").
		Raw("SUB $1, R2, R2").
		Raw("CBNZ R2, loop").
		Label("done").
		Ret()
	f.Add(b.Func())

	// The integer NEON multiply used here (VUMULL/VMUL) is only assemblable on Go
	// 1.27+, so the build constraint must match the //go:build go1.27 Go file;
	// emit writes a bare "arm64", which we narrow here.
	out := strings.Replace(f.String(), "//go:build arm64\n", "//go:build arm64 && go1.27\n", 1)
	if err := os.WriteFile("encode_arm64.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_arm64.s")
}
