//go:build ignore

// Command gen produces encode_riscv64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) encoder for riscv64 using the RVV vector extension, 4
// groups (16 input bytes -> 20 chars) per iteration.
//
// The kernel pins vl=4 at SEW=32 for the arithmetic and vl=16 at SEW=8 for the
// byte loads/shuffles/stores (VSETIVLI immediate vl); it therefore requires
// VLEN>=128. riscv64 is little-endian, so a per-lane byte reverse (VRGATHERVV at
// SEW=8 with a bswap index) turns each group's 4 bytes into the big-endian 32-bit
// value v the encoder wants. The five base-85 digits are pulled out with five
// reciprocal-multiply steps using VMULHUVV (the 32-bit mulhi): v/85 ==
// (v*0xC0C0C0C1)>>32>>6 (exact over all 2^32 values, post-shift via VSRLVI $6).
// The remainder is v - q*85 via VMULVV (multiply low) + VSUBVV; '!' is added with
// VADDVX.
//
// VRGATHERVV writes 0 for any out-of-range index, giving PSHUFB-style gap zeroing
// for free. Each digit byte sits in the low byte of its word lane (byte address
// 4l on little-endian), so the scatter packs D0..D3 at stride 4 (VSLLVI + VORVV)
// and expands to stride 5 with two overlapping 16-byte windows (bytes 0..15 and
// 4..19), each gathered with one VRGATHERVV from the packed digits and one from
// the 5th digit; two VSE8V stores (dst and dst+4) write the 20 bytes.
//
// The all-zero "z" shortcut, the (n&3) leftover groups, and the trailing fragment
// are handled by the caller via the stdlib, so this kernel only ever sees
// non-zero groups and always writes exactly 5 chars per group.
//
// Run: GOWORK=off go run encode_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
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
	f := emit.NewFile("riscv64")

	mrec := f.Data("a85mrec", le4(0xC0C0C0C1))
	// Per-lane byte-reverse index (little-endian word -> big-endian value v).
	bswap := f.Data("a85bswap", []byte{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12})
	// Scatter gather-indices; the digit byte sits at byte address 4l. Gaps use an
	// out-of-range index so VRGATHERVV writes 0. The RVV spec zeroes a gather lane
	// only when its index >= VLMAX (not >= VL); on VLEN>128 hardware VLMAX at
	// SEW=8/LMUL=1 exceeds 16 (e.g. 32 on the 256-bit SpacemiT X60), so a sentinel
	// of 16 would instead read an undefined tail element (observed as 0xff) and
	// corrupt the output. Use 255, which is >= VLMAX for every VLEN <= 2040 bits
	// (i.e. all real RVV hardware), so the gap reliably reads as 0. Two overlapping
	// windows: A = bytes 0..15 (at dst), B = bytes 4..19 (at dst+4).
	const z = 255
	expA := f.Data("a85expA", []byte{0, 1, 2, 3, z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12})
	d4A := f.Data("a85d4A", []byte{z, z, z, z, 0, z, z, z, z, 4, z, z, z, z, 8, z})
	expB := f.Data("a85expB", []byte{z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12, 13, 14, 15, z})
	d4B := f.Data("a85d4B", []byte{0, z, z, z, z, 4, z, z, z, z, 8, z, z, z, z, 12})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := riscv64.NewFunc("encodeGroups", sig, 0)
	bld := b.LoadArg("dst_base", "X5").LoadArg("src_base", "X6").LoadArg("blocks", "X7").
		// Load constant/index vectors once (SEW=8, vl=16).
		Raw("VSETIVLI $16, E8, M1, TA, MA, X28").
		Raw("MOV $%s(SB), X29", bswap).Raw("VLE8V (X29), V20").
		Raw("MOV $%s(SB), X29", expA).Raw("VLE8V (X29), V21").
		Raw("MOV $%s(SB), X29", d4A).Raw("VLE8V (X29), V22").
		Raw("MOV $%s(SB), X29", expB).Raw("VLE8V (X29), V23").
		Raw("MOV $%s(SB), X29", d4B).Raw("VLE8V (X29), V24").
		// mrec at SEW=32, vl=4.
		Raw("VSETIVLI $4, E32, M1, TA, MA, X28").
		Raw("MOV $%s(SB), X29", mrec).Raw("VLE32V (X29), V25").
		Raw("MOV $85, X30").   // q*85 multiplier (scalar -> splat)
		Raw("VMVVX X30, V26"). // V26 = {85,85,85,85}
		Raw("MOV $33, X31").   // '!' offset
		Raw("BEQ X7, X0, done").
		Label("loop").
		// Load 16 src bytes and byte-reverse each lane (SEW=8).
		Raw("VSETIVLI $16, E8, M1, TA, MA, X28").
		Raw("VLE8V (X6), V0").
		Raw("VRGATHERVV V20, V0, V1"). // V1 = bswapped bytes
		// Arithmetic at SEW=32, vl=4.
		Raw("VSETIVLI $4, E32, M1, TA, MA, X28")

	// digit: q = vReg/85 -> qReg ; remainder vReg - q*85 -> rReg. Scratch V16.
	digit := func(vReg, qReg, rReg string) {
		bld.Raw("VMULHUVV V25, %s, V16", vReg). // high32(v*mrec)
							Raw("VSRLVI $6, V16, %s", qReg).  // q = >>6
							Raw("VMULVV V26, %s, V16", qReg). // q*85
							Raw("VSUBVV V16, %s, %s", vReg, rReg)
	}

	digit("V1", "V2", "V6") // q1=V2, r4=V6
	digit("V2", "V3", "V5") // q2=V3, r3=V5
	digit("V3", "V4", "V7") // q3=V4, r2=V7
	digit("V4", "V8", "V9") // q4=V8 (=r0), r1=V9
	// D0=V8, D1=V9, D2=V7, D3=V5, D4=V6 ; digit at low byte of lane (address 4l).
	bld.
		// add '!' (33).
		Raw("VADDVX X31, V8, V8").
		Raw("VADDVX X31, V9, V9").
		Raw("VADDVX X31, V7, V7").
		Raw("VADDVX X31, V5, V5").
		Raw("VADDVX X31, V6, V6").
		// pack4 (stride 4): byte 4l+j = Dj. On little-endian the digit is at byte 4l,
		// so shift Dj left by 8*j bits (within the lane, j<=3) and OR.
		Raw("VSLLVI $8, V9, V9").
		Raw("VSLLVI $16, V7, V7").
		Raw("VSLLVI $24, V5, V5").
		Raw("VORVV V9, V8, V8").
		Raw("VORVV V7, V8, V8").
		Raw("VORVV V5, V8, V8"). // V8 = packed4 (D0..D3, stride 4)
		// scatter into two overlapping 16-byte windows (SEW=8). V8=packed4, V6=D4.
		Raw("VSETIVLI $16, E8, M1, TA, MA, X28").
		Raw("VRGATHERVV V21, V8, V10"). // window A: exp (gaps -> 0)
		Raw("VRGATHERVV V22, V6, V11"). // window A: d4
		Raw("VORVV V11, V10, V10").     // V10 = output bytes 0..15
		Raw("VRGATHERVV V23, V8, V12"). // window B: exp
		Raw("VRGATHERVV V24, V6, V13"). // window B: d4
		Raw("VORVV V13, V12, V12").     // V12 = output bytes 4..19
		Raw("VSE8V V10, (X5)").
		Raw("ADD $4, X5").
		Raw("VSE8V V12, (X5)").
		Raw("ADD $16, X5").
		Raw("ADD $16, X6").
		Raw("ADD $-1, X7").
		Raw("BNE X7, X0, loop").
		Label("done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("encode_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_riscv64.s")
}
