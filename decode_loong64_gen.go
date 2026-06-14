//go:build ignore

// Command gen produces decode_loong64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) decoder for loong64 using the LSX vector facility, 4
// groups (20 chars -> 16 bytes) per iteration.
//
// Each word lane l decodes one 5-char group with a base-85 multiply-accumulate
// (((c0*85+c1)*85+c2)*85+c3)*85+c4 (VMULW multiply-low word + VADDW), the inverse
// of the encoder's reciprocal-divide; the low 32 bits are exact for any valid
// encoding.
//
// Gather: the five chars of lane l sit at byte addresses 5l+k (k=0..4). For each
// k, a VMOVQ of src+k loads bytes k..k+15 and a single shared VSHUFB control
// {0,_,_,_,5,_,_,_,10,_,_,_,15,_,_,_} places window byte 5l into the LOW byte of
// word lane l (byte address 4l, little-endian); gaps use index 16, which selects
// the zero high source (VSHUFB Vctrl,Vlow,Vhigh,Vd: idx<16 -> Vlow[idx], idx in
// 16..31 -> Vhigh[idx-16]; Vhigh=0). So lane l's word equals the char; '!' (33)
// is subtracted with VSUBW.
//
// Output: after the MAC, word lane l holds v with byte at address 4l = v (LSB);
// a per-lane byte-reverse VSHUFB lays it out big-endian, then one VMOVQ writes
// the 16 bytes.
//
// The caller hands the kernel only clean runs of whole 5-char groups of regular
// base-85 digits ('!'..'u'); whitespace, "z", the short/invalid trailing group,
// flush, and CorruptInputError offsets are handled by the caller via the stdlib.
//
// Run: GOWORK=off go run decode_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
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
	f := emit.NewFile("loong64")

	c33 := f.Data("d85c33", le4(33))
	c85 := f.Data("d85c85", le4(85))
	// Gather control: window byte 5l -> word lane l low byte (addr 4l). Gaps = 16
	// -> zero (VSHUFB high source).
	const z = 16
	gath := f.Data("d85gath", []byte{0, z, z, z, 5, z, z, z, 10, z, z, z, 15, z, z, z})
	// Per-lane byte-reverse control for the big-endian output store.
	bswap := f.Data("d85bswap", []byte{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := loong64.NewFunc("decodeGroups", sig, 0)
	b.LoadArg("dst_base", "R4").LoadArg("src_base", "R5").LoadArg("blocks", "R6").
		Raw("MOVV $%s(SB), R7", c33).Raw("VMOVQ (R7), V22").
		Raw("MOVV $%s(SB), R7", c85).Raw("VMOVQ (R7), V21").
		Raw("MOVV $%s(SB), R7", gath).Raw("VMOVQ (R7), V23").
		Raw("MOVV $%s(SB), R7", bswap).Raw("VMOVQ (R7), V24").
		Raw("VXORV V28, V28, V28"). // zero (VSHUFB high source)
		Raw("BEQ R6, R0, done").
		Label("loop").
		// Gather C0..C4 from windows src+k via VSHUFB, subtract '!'.
		Raw("VMOVQ (R5), V0").Raw("VSHUFB V23, V0, V28, V1").Raw("VSUBW V22, V1, V1"). // C0
		Raw("ADDV $1, R5, R7").Raw("VMOVQ (R7), V0").Raw("VSHUFB V23, V0, V28, V2").Raw("VSUBW V22, V2, V2"). // C1
		Raw("ADDV $2, R5, R7").Raw("VMOVQ (R7), V0").Raw("VSHUFB V23, V0, V28, V3").Raw("VSUBW V22, V3, V3"). // C2
		Raw("ADDV $3, R5, R7").Raw("VMOVQ (R7), V0").Raw("VSHUFB V23, V0, V28, V4").Raw("VSUBW V22, V4, V4"). // C3
		Raw("ADDV $4, R5, R7").Raw("VMOVQ (R7), V0").Raw("VSHUFB V23, V0, V28, V5").Raw("VSUBW V22, V5, V5"). // C4
		// MAC: v = ((((C0*85+C1)*85+C2)*85+C3)*85+C4.
		Raw("VMULW V21, V1, V1").Raw("VADDW V2, V1, V1").
		Raw("VMULW V21, V1, V1").Raw("VADDW V3, V1, V1").
		Raw("VMULW V21, V1, V1").Raw("VADDW V4, V1, V1").
		Raw("VMULW V21, V1, V1").Raw("VADDW V5, V1, V1").
		// Big-endian byte reverse + store.
		Raw("VSHUFB V24, V1, V28, V0").
		Raw("VMOVQ V0, (R4)").
		Raw("ADDV $20, R5, R5").Raw("ADDV $16, R4, R4").Raw("ADDV $-1, R6, R6").
		Raw("BNE R6, R0, loop").
		Label("done").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("decode_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_loong64.s")
}
