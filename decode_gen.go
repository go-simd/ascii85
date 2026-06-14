//go:build ignore

// Command gen produces decode_amd64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) decoder, SSE path, 4 groups (20 chars -> 16 bytes) per
// iteration. Each 32-bit lane decodes one 5-char group with a base-85
// multiply-accumulate (((c0*85+c1)*85+c2)*85+c3)*85+c4, the inverse of the
// encoder's reciprocal-divide.
//
// Gather: the five characters of lane l sit at byte addresses 5l+k (k=0..4). For
// each k, an unaligned load of src+k brings bytes k..k+15 into a register, and a
// single shared PSHUFB control {0,_,_,_,5,_,_,_,10,_,_,_,15,_,_,_} (0x80 zeroing
// the gaps) pulls the four group chars into the low byte of each 32-bit lane. So
// five MOVOU+PSHUFB pairs build C0..C4. '!' (33) is subtracted, the MAC runs with
// PMULLD/PADDL (the low 32 bits are exact for any valid encoding), and a per-lane
// byte-reverse PSHUFB lays the value out big-endian for a single 16-byte store.
//
// The caller hands the kernel only clean runs of whole 5-char groups whose bytes
// are all regular base-85 digits ('!'..'u'); whitespace, the "z" shortcut, the
// short/invalid trailing group, flush, and CorruptInputError offsets are handled
// by the caller via the stdlib, so the kernel only ever sees valid groups and
// always writes exactly 4 bytes each.
//
// Run: GOWORK=off go run decode_gen.go
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

	// Gather control: lane l (32-bit) low byte <- window byte 5l; gaps -> 0.
	gath := f.Data("d85gath", []byte{0, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 15, 0x80, 0x80, 0x80})
	c33 := f.Data("d85c33", rep4(33))
	c85 := f.Data("d85c85", rep4(85))
	// Per-lane big-endian byte layout for the output store.
	bswap := f.Data("d85bswap", []byte{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	s := amd64.NewFunc("decodeGroups", sig, 0)
	s.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("blocks", "CX").
		Raw("MOVOU %s+0(SB), X8", gath).
		Raw("MOVOU %s+0(SB), X9", c33).
		Raw("MOVOU %s+0(SB), X10", c85).
		Raw("MOVOU %s+0(SB), X11", bswap).
		Raw("TESTQ CX, CX").Raw("JZ done").
		Label("blkloop").
		// Gather C0..C4: window src+k, PSHUFB, subtract '!'. Accumulate into X0.
		Raw("MOVOU 0(SI), X0").Raw("PSHUFB X8, X0").Raw("PSUBL X9, X0"). // X0 = C0
		Raw("MOVOU 1(SI), X1").Raw("PSHUFB X8, X1").Raw("PSUBL X9, X1"). // X1 = C1
		Raw("MOVOU 2(SI), X2").Raw("PSHUFB X8, X2").Raw("PSUBL X9, X2"). // X2 = C2
		Raw("MOVOU 3(SI), X3").Raw("PSHUFB X8, X3").Raw("PSUBL X9, X3"). // X3 = C3
		Raw("MOVOU 4(SI), X4").Raw("PSHUFB X8, X4").Raw("PSUBL X9, X4"). // X4 = C4
		// MAC: v = ((((C0*85+C1)*85+C2)*85+C3)*85+C4.
		Raw("PMULLD X10, X0").Raw("PADDL X1, X0").
		Raw("PMULLD X10, X0").Raw("PADDL X2, X0").
		Raw("PMULLD X10, X0").Raw("PADDL X3, X0").
		Raw("PMULLD X10, X0").Raw("PADDL X4, X0").
		// Big-endian layout per lane, store 16 bytes.
		Raw("PSHUFB X11, X0").
		Raw("MOVOU X0, (DI)").
		Raw("ADDQ $20, SI").Raw("ADDQ $16, DI").Raw("DECQ CX").Raw("JNZ blkloop").
		Label("done").Ret()
	f.Add(s.Func())

	if err := os.WriteFile("decode_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_amd64.s")
}
