//go:build ignore

// Command gen produces encode_loong64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) encoder for loong64 using the LSX vector facility, 4
// groups (16 input bytes -> 20 chars) per iteration.
//
// loong64 is little-endian, so after a VMOVQ load each word lane's value is byte-
// reversed relative to the big-endian 32-bit value v the encoder wants; a VSHUFB
// with a per-lane byte-reverse control fixes that. The five base-85 digits are
// pulled out with five reciprocal-multiply steps using VMUHWU (vector multiply
// high word unsigned = the 32-bit mulhi in a single op): v/85 ==
// (v*0xC0C0C0C1)>>32>>6 (exact over all 2^32 values, post-shift via VSRLW $6).
// The remainder is v - q*85 via VMULW (multiply low word) + VSUBW.
//
// LSX VSHUFB Vctrl, Vlow, Vhigh, Vd selects, per result byte, Vlow[idx] for
// idx 0..15 and Vhigh[idx-16] for idx 16..31 (verified under qemu). With Vhigh
// set to a zero vector, control index 16 produces a zero byte, giving PSHUFB-like
// gaps. Each digit byte lands in the low byte of its word lane (byte address
// 4l+0 on little-endian), so the scatter packs D0..D3 at stride 4 and expands to
// stride 5 with two overlapping 16-byte windows (bytes 0..15 and 4..19), each
// built by one VSHUFB from the packed digits and one from the 5th digit; two
// VMOVQ stores (dst and dst+4) write the 20 bytes.
//
// The all-zero "z" shortcut, the (n&3) leftover groups, and the trailing fragment
// are handled by the caller via the stdlib, so this kernel only ever sees
// non-zero groups and always writes exactly 5 chars per group.
//
// Run: GOWORK=off go run encode_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

// le4 stores a 32-bit constant little-endian into all four word lanes so the
// in-register word value (loong64 is LE) equals v.
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
	f := emit.NewFile("loong64")

	mrec := f.Data("a85mrec", le4(0xC0C0C0C1))
	c85 := f.Data("a85c85", le4(85))
	c33 := f.Data("a85c33", le4(33))
	// Per-lane byte-reverse control (little-endian word -> big-endian value).
	bswap := f.Data("a85bswap", []byte{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12})
	// Scatter controls. Each digit byte sits in the low byte of its word lane,
	// i.e. byte address 4l (little-endian). Gaps use index 16 -> zero (Vhigh=0).
	// Two overlapping windows: A = bytes 0..15 (at dst), B = bytes 4..19 (at dst+4).
	const z = 16
	expA := f.Data("a85expA", []byte{0, 1, 2, 3, z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12})
	d4A := f.Data("a85d4A", []byte{z, z, z, z, 0, z, z, z, z, 4, z, z, z, z, 8, z})
	expB := f.Data("a85expB", []byte{z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12, 13, 14, 15, z})
	d4B := f.Data("a85d4B", []byte{0, z, z, z, z, 4, z, z, z, z, 8, z, z, z, z, 12})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := loong64.NewFunc("encodeGroups", sig, 0)
	bld := b.LoadArg("dst_base", "R4").LoadArg("src_base", "R5").LoadArg("blocks", "R6").
		Raw("MOVV $%s(SB), R7", mrec).Raw("VMOVQ (R7), V20").
		Raw("MOVV $%s(SB), R7", c85).Raw("VMOVQ (R7), V21").
		Raw("MOVV $%s(SB), R7", c33).Raw("VMOVQ (R7), V22").
		Raw("MOVV $%s(SB), R7", bswap).Raw("VMOVQ (R7), V23").
		Raw("MOVV $%s(SB), R7", expA).Raw("VMOVQ (R7), V24").
		Raw("MOVV $%s(SB), R7", d4A).Raw("VMOVQ (R7), V25").
		Raw("MOVV $%s(SB), R7", expB).Raw("VMOVQ (R7), V26").
		Raw("MOVV $%s(SB), R7", d4B).Raw("VMOVQ (R7), V27").
		Raw("VXORV V28, V28, V28"). // zero (scatter gap source / VSHUFB high source)
		Raw("BEQ R6, R0, done").
		Label("loop").
		Raw("VMOVQ (R5), V0").
		Raw("VSHUFB V23, V0, V28, V0") // V0 = per-lane big-endian value v

	// digit: q = vReg/85 -> qReg ; remainder vReg - q*85 -> rReg. Scratch V16.
	digit := func(vReg, qReg, rReg string) {
		bld.Raw("VMUHWU V20, %s, V16", vReg). // V16 = high32(v*mrec)
							Raw("VSRLW $6, V16, %s", qReg).  // q = V16 >> 6
							Raw("VMULW V21, %s, V16", qReg). // V16 = q*85 (low word)
							Raw("VSUBW V16, %s, %s", vReg, rReg)
	}

	digit("V0", "V1", "V5") // q1=V1, r4=V5
	digit("V1", "V2", "V4") // q2=V2, r3=V4
	digit("V2", "V3", "V6") // q3=V3, r2=V6
	digit("V3", "V7", "V8") // q4=V7 (=r0), r1=V8
	// D0=V7, D1=V8, D2=V6, D3=V4, D4=V5 ; digit byte at low byte of word (address 4l).
	bld.
		// add '!' (33) to every digit.
		Raw("VADDW V22, V7, V7").
		Raw("VADDW V22, V8, V8").
		Raw("VADDW V22, V6, V6").
		Raw("VADDW V22, V4, V4").
		Raw("VADDW V22, V5, V5").
		// pack4 (stride 4): byte 4l+j = Dj. On little-endian the digit is already at
		// byte 4l, so shift Dj LEFT (toward higher addresses) by j bytes = VSLLW per
		// word by 8*j bits (stays within the 4-byte lane: j<=3).
		Raw("VSLLW $8, V8, V8").  // D1 byte -> 4l+1
		Raw("VSLLW $16, V6, V6"). // D2 byte -> 4l+2
		Raw("VSLLW $24, V4, V4"). // D3 byte -> 4l+3
		Raw("VORV V8, V7, V7").
		Raw("VORV V6, V7, V7").
		Raw("VORV V4, V7, V7"). // V7 = packed4 (D0..D3, stride 4)
		// expand stride-4 -> stride-5 into two overlapping windows; merge D4 (=V5).
		Raw("VSHUFB V24, V7, V28, V9").  // window A: exp
		Raw("VSHUFB V25, V5, V28, V10"). // window A: d4
		Raw("VORV V10, V9, V9").         // V9 = output bytes 0..15
		Raw("VSHUFB V26, V7, V28, V11"). // window B: exp
		Raw("VSHUFB V27, V5, V28, V12"). // window B: d4
		Raw("VORV V12, V11, V11").       // V11 = output bytes 4..19
		Raw("VMOVQ V9, (R4)").           // store bytes 0..15 at dst
		Raw("ADDV $4, R4, R4").
		Raw("VMOVQ V11, (R4)"). // store bytes 4..19 at dst+4
		Raw("ADDV $16, R4, R4").
		Raw("ADDV $16, R5, R5").
		Raw("ADDV $-1, R6, R6").
		Raw("BNE R6, R0, loop").
		Label("done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("encode_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_loong64.s")
}
