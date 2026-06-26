//go:build ignore

// Command gen produces decode_riscv64.s with go-asmgen: a vectorised ascii85
// (Adobe/btoa base-85) decoder for riscv64 using the RVV vector extension, 4
// groups (20 chars -> 16 bytes) per iteration.
//
// The kernel pins vl=4 at SEW=32 for the arithmetic and vl=16 at SEW=8 for the
// byte gathers/store (VSETIVLI immediate vl); it therefore requires VLEN>=128.
// Each word lane l decodes one 5-char group with a base-85 multiply-accumulate
// (((c0*85+c1)*85+c2)*85+c3)*85+c4 (VMULVV multiply-low + VADDVV), the inverse of
// the encoder's reciprocal-divide; the low 32 bits are exact for any valid
// encoding.
//
// Gather: the five chars of lane l sit at byte addresses 5l+k (k=0..4). For each
// k, a VLE8V of src+k loads bytes k..k+15 and a single shared VRGATHERVV index
// {0,_,_,_,5,_,_,_,10,_,_,_,15,_,_,_} places window byte 5l into the LOW byte of
// word lane l (byte address 4l, little-endian); gaps use an out-of-range index
// (255, >= VLMAX on all real hardware) which VRGATHERVV writes as 0. So lane l's
// word equals the char; '!' (33) is subtracted with VSUBVX.
//
// Output: after the MAC, word lane l holds v with byte at address 4l = v (LSB);
// a per-lane byte reverse (VRGATHERVV at SEW=8, index {3,2,1,0,...}) lays it out
// big-endian, then one VSE8V writes the 16 bytes.
//
// The caller hands the kernel only clean runs of whole 5-char groups of regular
// base-85 digits ('!'..'u'); whitespace, "z", the short/invalid trailing group,
// flush, and CorruptInputError offsets are handled by the caller via the stdlib.
//
// Run: GOWORK=off go run decode_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")

	// Gather index: window byte 5l -> word lane l low byte (addr 4l). Gaps use an
	// out-of-range index so VRGATHERVV writes 0. The RVV spec zeroes a gather lane
	// only when its index >= VLMAX (not >= VL); on VLEN>128 hardware VLMAX at
	// SEW=8/LMUL=1 exceeds 16 (e.g. 32 on the 256-bit SpacemiT X60), so a sentinel
	// of 16 would instead read an undefined tail element (observed as 0xff) and
	// corrupt the output. Use 255, which is >= VLMAX for every VLEN <= 2040 bits
	// (i.e. all real RVV hardware), so the gap reliably reads as 0.
	const z = 255
	gath := f.Data("d85gath", []byte{0, z, z, z, 5, z, z, z, 10, z, z, z, 15, z, z, z})
	// Per-lane byte-reverse index for the big-endian output store.
	bswap := f.Data("d85bswap", []byte{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12})

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)

	b := riscv64.NewFunc("decodeGroups", sig, 0)
	b.LoadArg("dst_base", "X5").LoadArg("src_base", "X6").LoadArg("blocks", "X7").
		// Load index vectors once (SEW=8, vl=16).
		Raw("VSETIVLI $16, E8, M1, TA, MA, X28").
		Raw("MOV $%s(SB), X29", gath).Raw("VLE8V (X29), V20").
		Raw("MOV $%s(SB), X29", bswap).Raw("VLE8V (X29), V21").
		Raw("MOV $33, X30"). // '!' offset
		Raw("MOV $85, X31"). // MAC multiplier
		Raw("BEQ X7, X0, done").
		Label("loop").
		// Gather C0..C4 from windows src+k (SEW=8), then subtract '!' at SEW=32.
		Raw("VSETIVLI $16, E8, M1, TA, MA, X28").
		Raw("VLE8V (X6), V0").Raw("VRGATHERVV V20, V0, V1"). // C0 raw
		Raw("ADD $1, X6, X29").Raw("VLE8V (X29), V0").Raw("VRGATHERVV V20, V0, V2").
		Raw("ADD $2, X6, X29").Raw("VLE8V (X29), V0").Raw("VRGATHERVV V20, V0, V3").
		Raw("ADD $3, X6, X29").Raw("VLE8V (X29), V0").Raw("VRGATHERVV V20, V0, V4").
		Raw("ADD $4, X6, X29").Raw("VLE8V (X29), V0").Raw("VRGATHERVV V20, V0, V5").
		Raw("VSETIVLI $4, E32, M1, TA, MA, X28").
		Raw("VSUBVX X30, V1, V1").
		Raw("VSUBVX X30, V2, V2").
		Raw("VSUBVX X30, V3, V3").
		Raw("VSUBVX X30, V4, V4").
		Raw("VSUBVX X30, V5, V5").
		// MAC: v = ((((C0*85+C1)*85+C2)*85+C3)*85+C4.
		Raw("VMULVX X31, V1, V1").Raw("VADDVV V2, V1, V1").
		Raw("VMULVX X31, V1, V1").Raw("VADDVV V3, V1, V1").
		Raw("VMULVX X31, V1, V1").Raw("VADDVV V4, V1, V1").
		Raw("VMULVX X31, V1, V1").Raw("VADDVV V5, V1, V1").
		// Big-endian byte reverse + store (SEW=8).
		Raw("VSETIVLI $16, E8, M1, TA, MA, X28").
		Raw("VRGATHERVV V21, V1, V0").
		Raw("VSE8V V0, (X5)").
		Raw("ADD $20, X6").Raw("ADD $16, X5").Raw("ADD $-1, X7").
		Raw("BNE X7, X0, loop").
		Label("done").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("decode_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_riscv64.s")
}
