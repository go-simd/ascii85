//go:build ignore

// Command gen produces encode_ppc64x.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) encoder for ppc64le using the VSX/AltiVec vector facility,
// 4 groups (16 input bytes -> 20 chars) per iteration.
//
// A 16-byte LXVB16X load (byte-order-correct on little-endian; LXVD2X would swap
// doublewords) places each group's 4 bytes into a word lane whose in-register
// big-endian interpretation is exactly the 32-bit value v the encoder wants, so
// no byte-reversal is needed. The five base-85 digits are pulled out with five
// reciprocal-multiply steps: the 32-bit mulhi is VMULEUW (even-lane 32x32->64
// products, high32 at words 0,2) and VMULOUW (odd lanes) merged by VMRGEW into
// [h0,h1,h2,h3], then VSRW by 6 (v/85 == (v*0xC0C0C0C1)>>32>>6, exact over all
// 2^32 values). The remainder is v - q*85 via VMULUWM (32x32->low32) + VSUBUWM.
//
// Each digit byte lands in the low byte of its word lane, i.e. byte address 4l+3
// (big-endian). VSLW by 24 moves it to byte address 4l (the lane's lowest /
// most-significant address), so the cross-lane scatter uses byte-address VPERM
// controls identical to the s390x ones. The scatter packs D0..D3 at stride 4
// (VSLDOI byte shifts + VOR), expands to stride 5, and merges the 5th digit;
// STXVB16X stores 16 bytes and STXSIWX the last word (bytes 16..19).
//
// VSX<->VMX aliasing: vectors are loaded with LXVB16X into VS(32+k) and operated
// on as Vk (the AltiVec name aliasing VS(32+k)); stores use STXVB16X/STXSIWX of
// VS(32+k).
//
// The all-zero "z" shortcut, the (n&3) leftover groups, and the trailing fragment
// are handled by the caller via the stdlib, so this kernel only ever sees
// non-zero groups and always writes exactly 5 chars per group.
//
// Run: GOWORK=off go run encode_ppc64x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

// be4 stores a 32-bit constant big-endian into all four word lanes so the
// in-register big-endian word value (after an LXVB16X load) equals v.
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

	mrec := f.Data("a85mrec", be4(0xC0C0C0C1))
	c85 := f.Data("a85c85", be4(85))
	c33 := f.Data("a85c33", be4(33))
	c24 := f.Data("a85c24", be4(24)) // VSLW shift count to move byte 4l+3 -> 4l
	c6 := f.Data("a85c6", be4(6))    // VSRW shift count for /85 post-shift
	// Scatter controls (byte-address indexed). Gaps use index 16, which selects
	// byte 0 of the all-zero second VPERM operand (ppc VPERM has no 0x80-zeroing).
	// Two overlapping 16-byte windows cover the 20 output bytes: window A = bytes
	// 0..15 (stored at dst), window B = bytes 4..19 (stored at dst+4). Storing 16
	// bytes twice (overlapping bytes 4..15) avoids a fiddly 4-byte tail store and
	// stays within the block's 20 bytes.
	const z = 16
	expA := f.Data("a85expA", []byte{0, 1, 2, 3, z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12})
	d4A := f.Data("a85d4A", []byte{z, z, z, z, 0, z, z, z, z, 4, z, z, z, z, 8, z})
	expB := f.Data("a85expB", []byte{z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12, 13, 14, 15, z})
	d4B := f.Data("a85d4B", []byte{0, z, z, z, z, 4, z, z, z, z, 8, z, z, z, z, 12})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := ppc64.NewFunc("encodeGroups", sig, 0)
	bld := b.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("blocks", "R5").
		// Load constant tables (LXVB16X into VS(32+k); used as Vk).
		Raw("MOVD $%s(SB), R6", mrec).Raw("LXVB16X (R6)(R0), VS52"). // V20 = mrec
		Raw("MOVD $%s(SB), R6", c85).Raw("LXVB16X (R6)(R0), VS53").  // V21 = 85
		Raw("MOVD $%s(SB), R6", c33).Raw("LXVB16X (R6)(R0), VS54").  // V22 = 33
		Raw("MOVD $%s(SB), R6", c24).Raw("LXVB16X (R6)(R0), VS55").  // V23 = 24
		Raw("MOVD $%s(SB), R6", c6).Raw("LXVB16X (R6)(R0), VS56").   // V24 = 6
		Raw("MOVD $%s(SB), R6", expA).Raw("LXVB16X (R6)(R0), VS57"). // V25 = exp A (bytes 0..15)
		Raw("MOVD $%s(SB), R6", expB).Raw("LXVB16X (R6)(R0), VS58"). // V26 = exp B (bytes 4..19)
		Raw("MOVD $%s(SB), R6", d4A).Raw("LXVB16X (R6)(R0), VS59").  // V27 = d4 A
		Raw("MOVD $%s(SB), R6", d4B).Raw("LXVB16X (R6)(R0), VS60").  // V28 = d4 B
		Raw("VSPLTISW $0, V29").                                     // V29 = zero (scatter gap source)
		Raw("CMP R5, $0").Raw("BEQ done").
		Label("loop").
		Raw("LXVB16X (R4)(R0), VS32") // V0 = four big-endian word values v

	// digit: q = vReg/85 -> qReg ; remainder vReg - q*85 -> rReg. Scratch V16,V17,V18.
	digit := func(vReg, qReg, rReg string) {
		bld.Raw("VMULEUW %s, V20, V16", vReg).
			Raw("VMULOUW %s, V20, V17", vReg).
			Raw("VMRGEW V16, V17, V18"). // [h0,h1,h2,h3]
			Raw("VSRW V18, V24, %s", qReg).
			Raw("VMULUWM %s, V21, V16", qReg). // q*85 (low32)
			Raw("VSUBUWM %s, V16, %s", vReg, rReg)
	}

	digit("V0", "V1", "V5") // q1=V1, r4=V5
	digit("V1", "V2", "V4") // q2=V2, r3=V4
	digit("V2", "V3", "V6") // q3=V3, r2=V6
	digit("V3", "V7", "V8") // q4=V7 (=r0), r1=V8
	// D0=V7, D1=V8, D2=V6, D3=V4, D4=V5
	bld.
		// add '!' (33) to every digit.
		Raw("VADDUWM V7, V22, V7").
		Raw("VADDUWM V8, V22, V8").
		Raw("VADDUWM V6, V22, V6").
		Raw("VADDUWM V4, V22, V4").
		Raw("VADDUWM V5, V22, V5").
		// move each digit from byte 4l+3 to byte 4l (VSLW left by 24 bits per word).
		Raw("VSLW V7, V23, V7").
		Raw("VSLW V8, V23, V8").
		Raw("VSLW V6, V23, V6").
		Raw("VSLW V4, V23, V4").
		Raw("VSLW V5, V23, V5").
		// pack4 (stride 4): byte 4l+j = Dj. Shift Dj toward higher byte addresses by
		// j bytes. VSLDOI $(16-j), Vzero, Dj => Dj >> j bytes (big-endian).
		Raw("VSLDOI $15, V29, V8, V9").  // D1 >> 1
		Raw("VSLDOI $14, V29, V6, V10"). // D2 >> 2
		Raw("VSLDOI $13, V29, V4, V11"). // D3 >> 3
		Raw("VOR V9, V7, V7").
		Raw("VOR V10, V7, V7").
		Raw("VOR V11, V7, V7"). // V7 = packed4 (D0..D3, stride 4)
		// expand stride-4 -> stride-5 and merge D4 (=V5) into both windows.
		Raw("VPERM V7, V29, V25, V12"). // window A: exp
		Raw("VPERM V5, V29, V27, V13"). // window A: d4
		Raw("VOR V13, V12, V12").       // V12 = output bytes 0..15
		Raw("VPERM V7, V29, V26, V14"). // window B: exp
		Raw("VPERM V5, V29, V28, V15"). // window B: d4
		Raw("VOR V15, V14, V14").       // V14 = output bytes 4..19
		Raw("STXVB16X VS44, (R3)(R0)"). // store bytes 0..15 at dst
		Raw("ADD $4, R3").
		Raw("STXVB16X VS46, (R3)(R0)"). // store bytes 4..19 at dst+4
		Raw("ADD $16, R3").
		Raw("ADD $16, R4").
		Raw("ADD $-1, R5").
		Raw("CMP R5, $0").Raw("BNE loop").
		Label("done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("encode_ppc64x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_ppc64x.s")
}
