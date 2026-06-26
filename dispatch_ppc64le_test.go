//go:build ppc64le

package ascii85

import (
	"bytes"
	stdascii85 "encoding/ascii85"
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestDispatchPPC64LE drives both the encode and decode kernels down both of
// their branches — the VSX kernel and the scalar loop fallback (encodeScalar /
// decodeScalar). The kernels emit ISA-3.0 (POWER9) instructions
// (LXVB16X/STXVB16X) that raise SIGILL on POWER8, so the kernel-forcing branch
// runs only when the host is actually POWER9+. The scalar-fallback branch is
// always exercised. The power9-targeted QEMU CI job and the native POWER9/POWER10
// farm runs cover the kernel branch. Each variant is checked byte-for-byte
// against encoding/ascii85 (encode) plus a full decode round trip.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

	rng := rand.New(rand.NewSource(11))
	check := func(label string) {
		for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 20, 21, 64, 100, 257, 1000} {
			src := make([]byte, n)
			rng.Read(src)
			// Avoid all-zero groups for some bytes but keep some too: the encoder
			// routes zero groups to stdlib regardless, so both code paths still run.
			dst := make([]byte, MaxEncodedLen(n))
			w := Encode(dst, src)
			got := dst[:w]

			wantBuf := make([]byte, stdascii85.MaxEncodedLen(n))
			want := wantBuf[:stdascii85.Encode(wantBuf, src)]
			if !bytes.Equal(got, want) {
				t.Fatalf("%s n=%d: Encode mismatch vs stdlib", label, n)
			}

			// Round trip: decode the encoded bytes and compare to src. The
			// destination must have room for whole 4-byte groups even for a
			// short final fragment: encoding/ascii85.Decode (which our Decode
			// mirrors) emits in 4-byte chunks and writes nothing for a trailing
			// group when fewer than 4 bytes of dst remain, so size out to
			// MaxDecodedLen rounded up to a 4-byte multiple.
			out := make([]byte, (n+3)/4*4+4)
			ndst, _, err := Decode(out, got, true)
			if err != nil {
				t.Fatalf("%s n=%d: Decode error %v", label, n, err)
			}
			if ndst != n || !bytes.Equal(out[:ndst], src) {
				t.Fatalf("%s n=%d: round trip mismatch (ndst=%d)", label, n, ndst)
			}
		}
	}

	// Scalar fallback: always safe on every ppc64le host (POWER8 included).
	hasVSX = false
	check("fallback")

	// VSX kernel: only force it on when the CPU is POWER9+, otherwise the
	// LXVB16X/STXVB16X in the kernels would SIGILL (e.g. on a POWER8 farm node).
	if !cpu.PPC64.IsPOWER9 {
		t.Log("CPU is pre-POWER9; VSX kernel branch not exercised on this host")
		return
	}
	hasVSX = true
	check("kernel")
}
