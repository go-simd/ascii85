//go:build ignore

// Command gen produces encode_s390x.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) encoder for s390x using the z/Architecture vector
// facility, 4 groups (16 input bytes -> 20 chars) per iteration.
//
// s390x is BIG-ENDIAN, which is convenient here: a plain VL of 16 bytes places
// lane l's fullword = src[4l..4l+3] interpreted big-endian, i.e. exactly the
// 32-bit value v the encoder wants, with no byte-reversal. The five base-85
// digits are pulled out with five reciprocal-multiply steps using VMLHF (vector
// multiply logical high fullword = the 32-bit mulhi in a single op):
// v/85 == (v*0xC0C0C0C1) high32 >> 6 (exact over all 2^32 values); the remainder
// is v - q*85 via VMLF (multiply low fullword) + VSF (subtract fullword).
//
// Each digit byte lands in the LOW byte of its fullword lane, i.e. byte address
// 4l+3 (big-endian). VESLF $24 shifts it to byte address 4l (the lane's lowest
// address), so the cross-lane scatter into the stride-5 output uses byte-address
// VPERM controls identical to the little-endian PSHUFB controls (VPERM and
// PSHUFB both index by byte address, which is endianness-independent). The
// scatter packs D0..D3 at stride 4 (VSLDB byte shifts + VO), expands to stride 5,
// and merges the 5th digit; VST stores 16 bytes and VSTEF the last 4.
//
// The all-zero "z" shortcut, the (n&3) leftover groups, and the trailing
// fragment are handled by the caller via the stdlib, so this kernel only ever
// sees non-zero groups and always writes exactly 5 chars per group.
//
// Run: GOWORK=off go run encode_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

// be4 stores a 32-bit constant big-endian into all four fullword lanes so the
// in-register value (as the fullword vector ops see it) equals v.
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

	mrec := f.Data("a85mrec", be4(0xC0C0C0C1))
	c85 := f.Data("a85c85", be4(85))
	c33 := f.Data("a85c33", be4(33))
	// Scatter controls (byte-address indexed). s390x VPERM has no PSHUFB-style
	// 0x80-zeroing: an out-of-the-first-operand index is taken modulo 32, so gaps
	// use index 16, which selects byte 0 of the all-zero second VPERM operand.
	const z = 16 // points at the zero vector -> output byte 0
	expLo := f.Data("a85expLo", []byte{0, 1, 2, 3, z, 4, 5, 6, 7, z, 8, 9, 10, 11, z, 12})
	expHi := f.Data("a85expHi", []byte{13, 14, 15, z, z, z, z, z, z, z, z, z, z, z, z, z})
	d4Lo := f.Data("a85d4Lo", []byte{z, z, z, z, 0, z, z, z, z, 4, z, z, z, z, 8, z})
	d4Hi := f.Data("a85d4Hi", []byte{z, z, z, 12, z, z, z, z, z, z, z, z, z, z, z, z})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	g := s390x.NewFunc("encodeGroups", sig, 0)
	bld := g.LoadArg("dst_base", "R1").LoadArg("src_base", "R2").LoadArg("blocks", "R3").
		Raw("MOVD $%s(SB), R4", mrec).Raw("VL (R4), V20").
		Raw("MOVD $%s(SB), R4", c85).Raw("VL (R4), V21").
		Raw("MOVD $%s(SB), R4", c33).Raw("VL (R4), V22").
		Raw("MOVD $%s(SB), R4", expLo).Raw("VL (R4), V23").
		Raw("MOVD $%s(SB), R4", expHi).Raw("VL (R4), V24").
		Raw("MOVD $%s(SB), R4", d4Lo).Raw("VL (R4), V25").
		Raw("MOVD $%s(SB), R4", d4Hi).Raw("VL (R4), V26").
		Raw("CMPBEQ R3, $0, done").
		Label("loop").
		Raw("VL (R2), V0") // V0 = four big-endian fullword values v

	// digit step: given value vReg, compute q = (vReg/85) into qReg, remainder
	// (vReg - q*85) into rReg, using V16,V17 scratch. VMLHF Va,Vb,Vt => Vt =
	// high32(Va*Vb); VESRLF $6 => /64-bias shift; VMLF => low32 product; VSF Vb,Va,Vt
	// => Vt = Va - Vb.
	digit := func(vReg, qReg, rReg string) {
		bld.Raw("VMLHF %s, V20, V16", vReg).
			Raw("VESRLF $6, V16, %s", qReg).
			Raw("VMLF %s, V21, V17", qReg).
			Raw("VSF V17, %s, %s", vReg, rReg)
	}

	digit("V0", "V1", "V5") // q1=V1, r4=V5
	digit("V1", "V2", "V4") // q2=V2, r3=V4
	digit("V2", "V3", "V6") // q3=V3, r2=V6
	digit("V3", "V7", "V8") // q4=V7 (=r0), r1=V8
	// rename for clarity: D0=V7, D1=V8, D2=V6, D3=V4, D4=V5
	bld.
		// add '!' (33) to every digit.
		Raw("VAF V22, V7, V7").
		Raw("VAF V22, V8, V8").
		Raw("VAF V22, V6, V6").
		Raw("VAF V22, V4, V4").
		Raw("VAF V22, V5, V5").
		// move each digit from byte 4l+3 to byte 4l (lane's lowest address).
		Raw("VESLF $24, V7, V7").
		Raw("VESLF $24, V8, V8").
		Raw("VESLF $24, V6, V6").
		Raw("VESLF $24, V4, V4").
		Raw("VESLF $24, V5, V5").
		// pack4 (stride 4): byte 4l+j = Dj's lane-l digit. On big-endian s390x this
		// means shifting Dj toward HIGHER byte addresses by j bytes. VSLDB $(16-j),
		// V19(zero), Dj => Dj shifted right by j bytes (verified under qemu).
		Raw("VZERO V19").
		Raw("VSLDB $15, V19, V8, V9").  // D1 >> 1 byte
		Raw("VSLDB $14, V19, V6, V10"). // D2 >> 2 bytes
		Raw("VSLDB $13, V19, V4, V11"). // D3 >> 3 bytes
		Raw("VO V9, V7, V7").
		Raw("VO V10, V7, V7").
		Raw("VO V11, V7, V7"). // V7 = packed4 (D0..D3, stride 4)
		// expand stride-4 -> stride-5 and merge D4 (=V5). VPERM Va,Vzero,Vctrl,Vt
		// selects from (Va||Vzero) by byte index; gap index 16 -> zero byte. V19 is 0.
		Raw("VPERM V7, V19, V23, V12"). // exp lo
		Raw("VPERM V5, V19, V25, V13"). // d4 lo
		Raw("VO V13, V12, V12").        // V12 = output bytes 0..15
		Raw("VPERM V7, V19, V24, V14"). // exp hi
		Raw("VPERM V5, V19, V26, V15"). // d4 hi
		Raw("VO V15, V14, V14").        // V14 = output bytes 16..19 in its low (first) 4 bytes
		Raw("VST V12, (R1)").
		Raw("VSTEF $0, V14, 16(R1)"). // store the first fullword (bytes 16..19)
		Raw("ADD $16, R2").
		Raw("ADD $20, R1").
		Raw("ADD $-1, R3").
		Raw("CMPBNE R3, $0, loop").
		Label("done").
		Ret()
	f.Add(g.Func())

	if err := os.WriteFile("encode_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_s390x.s")
}
