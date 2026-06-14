//go:build ignore

// Command gen produces encode_amd64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) encoder, SSE path, 4 groups (16 input bytes -> 20 chars)
// per iteration. Each 32-bit lane holds one 4-byte group as a big-endian value;
// the five base-85 digits are pulled out with five reciprocal-multiply steps
// (v/85 == (v*0xC0C0C0C1)>>32>>6, exact over all 2^32 values; remainder =
// v - q*85). The four 32-bit-lane mulhi values come from two PMULULQ (even/odd
// dwords) blended back together. The five digit vectors are then scattered into
// the stride-5 output layout with PSHUFB control vectors (a stride-4 pack via
// byte-shift ORs, expanded to stride 5, plus the 5th digit). Each call processes
// exactly blocks*4 groups; the all-zero "z" shortcut, the (n&3) leftover groups,
// and the trailing fragment are handled by the caller via the stdlib, so this
// kernel only ever sees non-zero groups and always writes 5 chars each.
//
// Run: GOWORK=off go run encode_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func rep4(v uint32) []byte {
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
	f := emit.NewFile("amd64")

	// Per-lane big-endian load: reverse bytes within each 32-bit lane.
	bswap := f.Data("a85bswap", []byte{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12})
	mrec := f.Data("a85mrec", rep4(0xC0C0C0C1)) // /85 reciprocal multiplier
	c85 := f.Data("a85c85", rep4(85))
	c33 := f.Data("a85c33", rep4(33))
	// Blend mask: 0xffffffff in dwords 1 and 3, 0 in dwords 0 and 2. Used to merge
	// the odd-lane mulhi (in dwords 1,3) with the even-lane mulhi (dwords 0,2).
	oddmask := f.Data("a85oddmask", []byte{
		0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff,
		0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff,
	})
	// Scatter controls (PSHUFB: out[i] = src[ctrl[i]] unless ctrl[i]&0x80).
	expLo := f.Data("a85expLo", []byte{0, 1, 2, 3, 0x80, 4, 5, 6, 7, 0x80, 8, 9, 10, 11, 0x80, 12})
	expHi := f.Data("a85expHi", []byte{13, 14, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80})
	d4Lo := f.Data("a85d4Lo", []byte{0x80, 0x80, 0x80, 0x80, 0, 0x80, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 0x80, 8, 0x80})
	d4Hi := f.Data("a85d4Hi", []byte{0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	s := amd64.NewFunc("encodeGroups", sig, 0)
	bld := s.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("blocks", "CX").
		Raw("MOVOU %s+0(SB), X8", bswap).
		Raw("MOVOU %s+0(SB), X9", mrec).
		Raw("MOVOU %s+0(SB), X10", c85).
		Raw("MOVOU %s+0(SB), X11", c33).
		Raw("MOVOU %s+0(SB), X12", oddmask).
		Raw("TESTQ CX, CX").Raw("JZ done")

	// mh: X5 = high32 of (srcReg * mrec) per 32-bit lane; scratch X6,X7.
	mh := func(srcReg string) {
		bld.Raw("MOVO %s, X6", srcReg).Raw("PMULULQ X9, X6").Raw("PSRLQ $32, X6"). // even dwords 0,2
												Raw("MOVO %s, X7", srcReg).Raw("PSRLO $4, X7").
												Raw("PMULULQ X9, X7").Raw("PSRLQ $32, X7").Raw("PSLLO $4, X7"). // odd dwords 1,3
												Raw("PAND X12, X7").                                            // keep dwords 1,3 of X7
												Raw("MOVO X12, X5").Raw("PANDN X6, X5").                        // X5 = ~mask & X6 (dwords 0,2)
												Raw("POR X7, X5")                                               // X5 = blended high32
	}

	bld.Label("blkloop").
		Raw("MOVOU (SI), X0").Raw("PSHUFB X8, X0") // X0 = v (big-endian per lane)
	mh("X0")
	bld.Raw("PSRLL $6, X5").Raw("MOVO X5, X1"). // X1 = q1 = v/85
							Raw("MOVO X1, X13").Raw("PMULLD X10, X13").Raw("MOVO X0, X4").Raw("PSUBL X13, X4") // X4 = r4 = v - q1*85
	mh("X1")
	bld.Raw("PSRLL $6, X5").Raw("MOVO X5, X2"). // X2 = q2
							Raw("MOVO X2, X13").Raw("PMULLD X10, X13").Raw("MOVO X1, X3").Raw("PSUBL X13, X3") // X3 = r3 = q1 - q2*85
	mh("X2")
	bld.Raw("PSRLL $6, X5").Raw("MOVO X5, X1"). // X1 = q3
							Raw("MOVO X1, X13").Raw("PMULLD X10, X13").Raw("PSUBL X13, X2") // X2 = r2 = q2 - q3*85
	mh("X1")
	bld.Raw("PSRLL $6, X5"). // X5 = q4 = r0
					Raw("MOVO X5, X13").Raw("PMULLD X10, X13").Raw("PSUBL X13, X1"). // X1 = r1 = q3 - q4*85
		// X5=r0 X1=r1 X2=r2 X3=r3 X4=r4 ; add '!' (33).
		Raw("PADDL X11, X5").Raw("PADDL X11, X1").Raw("PADDL X11, X2").Raw("PADDL X11, X3").Raw("PADDL X11, X4").
		// pack4 (stride 4): X5 | X1<<1 | X2<<2 | X3<<3 (byte shifts).
		Raw("MOVO X1, X6").Raw("PSLLO $1, X6").
		Raw("MOVO X2, X7").Raw("PSLLO $2, X7").
		Raw("MOVO X3, X13").Raw("PSLLO $3, X13").
		Raw("POR X6, X5").Raw("POR X7, X5").Raw("POR X13, X5"). // X5 = packed4 (D0..D3, stride 4)
		// expand stride-4 -> stride-5 lo/hi and merge the 5th digit (X4).
		Raw("MOVOU %s+0(SB), X6", expLo).Raw("MOVO X5, X7").Raw("PSHUFB X6, X7").
		Raw("MOVOU %s+0(SB), X6", d4Lo).Raw("MOVO X4, X13").Raw("PSHUFB X6, X13").
		Raw("POR X13, X7"). // X7 = output bytes 0..15
		Raw("MOVOU %s+0(SB), X6", expHi).Raw("MOVO X5, X0").Raw("PSHUFB X6, X0").
		Raw("MOVOU %s+0(SB), X6", d4Hi).Raw("PSHUFB X6, X4").
		Raw("POR X4, X0"). // X0 = output bytes 16..19 in its low 4 bytes
		Raw("MOVOU X7, (DI)").
		Raw("MOVL X0, 16(DI)").
		Raw("ADDQ $16, SI").Raw("ADDQ $20, DI").Raw("DECQ CX").Raw("JNZ blkloop").
		Label("done").Ret()
	f.Add(s.Func())

	if err := os.WriteFile("encode_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_amd64.s")
}
