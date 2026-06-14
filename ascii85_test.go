package ascii85

import (
	stdascii85 "encoding/ascii85"
	"math/rand"
	"testing"
)

// sizes exercises the tail (n&3), single/multi SSE blocks (4 groups), and large
// buffers, plus boundaries around the 16-input-byte block.
var sizes = []int{0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 17, 19, 20, 31, 32, 33, 64, 100, 1000, 4096, 65537}

func randBytes(seed int64, n int) []byte {
	b := make([]byte, n)
	rand.New(rand.NewSource(seed)).Read(b)
	return b
}

func TestEncode(t *testing.T) {
	for _, n := range sizes {
		src := randBytes(1, n)
		dst := make([]byte, MaxEncodedLen(n))
		got := dst[:Encode(dst, src)]

		wantBuf := make([]byte, stdascii85.MaxEncodedLen(n))
		want := wantBuf[:stdascii85.Encode(wantBuf, src)]
		if string(got) != string(want) {
			t.Fatalf("n=%d:\n got=%q\nwant=%q", n, got, want)
		}
	}
}

// TestEncodeZeroShortcut feeds data that is mostly all-zero groups (and zero
// groups interleaved with non-zero ones) so the "z" shortcut and the run-split
// in Encode are exercised and stay byte-identical to the stdlib.
func TestEncodeZeroShortcut(t *testing.T) {
	cases := [][]byte{
		make([]byte, 4),  // single zero group -> "z"
		make([]byte, 20), // five zero groups -> "zzzzz"
		make([]byte, 18), // four zero groups + 2-byte tail
		{0, 0, 0, 0, 1, 2, 3, 4},
		{1, 2, 3, 4, 0, 0, 0, 0},
		{1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8},
		{0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9},
		{0, 0, 0, 0, 0, 0, 0},    // zero group + 3-byte zero tail
		{0, 0, 0, 1, 0, 0, 0, 0}, // non-zero group then zero group
	}
	for i, src := range cases {
		dst := make([]byte, MaxEncodedLen(len(src)))
		got := dst[:Encode(dst, src)]
		want := make([]byte, stdascii85.MaxEncodedLen(len(src)))
		want = want[:stdascii85.Encode(want, src)]
		if string(got) != string(want) {
			t.Fatalf("case %d (%v):\n got=%q\nwant=%q", i, src, got, want)
		}
	}
}

func TestMaxEncodedLen(t *testing.T) {
	for _, n := range sizes {
		if got, want := MaxEncodedLen(n), stdascii85.MaxEncodedLen(n); got != want {
			t.Fatalf("MaxEncodedLen(%d)=%d want %d", n, got, want)
		}
	}
}

func TestDecode(t *testing.T) {
	for _, n := range sizes {
		src := randBytes(3, n)
		enc := make([]byte, MaxEncodedLen(n))
		enc = enc[:Encode(enc, src)]

		dst := make([]byte, 4*((len(enc)+4)/5)+8)
		ndst, nsrc, err := Decode(dst, enc, true)
		if err != nil {
			t.Fatalf("n=%d: Decode: %v", n, err)
		}
		if nsrc != len(enc) {
			t.Fatalf("n=%d: nsrc=%d want %d", n, nsrc, len(enc))
		}
		if string(dst[:ndst]) != string(src) {
			t.Fatalf("n=%d: round-trip mismatch", n)
		}
	}
}

// TestDecodeMatchesStdlib drives Decode (including whitespace, the "z" shortcut,
// and flush=false) and checks every return value matches encoding/ascii85.
func TestDecodeMatchesStdlib(t *testing.T) {
	inputs := []string{
		"",
		"<~z~>"[2:3],                // "z"
		"  z  ",                     // whitespace + z
		"87cURD]i,\"Ebo80",          // "Hello, Wor"
		"\t87cU\nRD]i, \"Ebo80\r\n", // whitespace interleaved
		"z!!!!!",                    // z then a zero-ish group
	}
	for _, flush := range []bool{false, true} {
		for i, in := range inputs {
			src := []byte(in)
			gotDst := make([]byte, 256)
			wantDst := make([]byte, 256)
			gnd, gns, gerr := Decode(gotDst, src, flush)
			wnd, wns, werr := stdascii85.Decode(wantDst, src, flush)
			if gnd != wnd || gns != wns || (gerr == nil) != (werr == nil) {
				t.Fatalf("case %d flush=%v: got (%d,%d,%v) want (%d,%d,%v)", i, flush, gnd, gns, gerr, wnd, wns, werr)
			}
			if string(gotDst[:gnd]) != string(wantDst[:wnd]) {
				t.Fatalf("case %d flush=%v: data mismatch", i, flush)
			}
		}
	}
}

func TestDecodeError(t *testing.T) {
	// A byte outside the valid range must yield a CorruptInputError at the right
	// offset, identical to the stdlib.
	src := []byte("87c\x00URD")
	dst := make([]byte, 32)
	_, _, err := Decode(dst, src, true)
	if err == nil {
		t.Fatal("Decode: want error on invalid input, got nil")
	}
	if _, ok := err.(CorruptInputError); !ok {
		t.Fatalf("want CorruptInputError, got %T", err)
	}
	_, _, werr := stdascii85.Decode(dst, src, true)
	if err.Error() != werr.Error() {
		t.Fatalf("error mismatch: got %v want %v", err, werr)
	}
}

func FuzzEncode(f *testing.F) {
	f.Add([]byte("hello world"))
	f.Add(make([]byte, 16))
	f.Add([]byte{1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8})
	f.Fuzz(func(t *testing.T, src []byte) {
		gotBuf := make([]byte, MaxEncodedLen(len(src)))
		got := gotBuf[:Encode(gotBuf, src)]
		wantBuf := make([]byte, stdascii85.MaxEncodedLen(len(src)))
		want := wantBuf[:stdascii85.Encode(wantBuf, src)]
		if string(got) != string(want) {
			t.Fatalf("got=%q want=%q", got, want)
		}
	})
}

func FuzzDecode(f *testing.F) {
	f.Add([]byte("87cURD]i,\"Ebo80"), true)
	f.Add([]byte("  z \t!!!!!"), false)
	f.Add([]byte("garbage!\x00"), true)
	f.Fuzz(func(t *testing.T, src []byte, flush bool) {
		gotDst := make([]byte, len(src)*4+16)
		wantDst := make([]byte, len(src)*4+16)
		gnd, gns, gerr := Decode(gotDst, src, flush)
		wnd, wns, werr := stdascii85.Decode(wantDst, src, flush)
		if gnd != wnd || gns != wns || (gerr == nil) != (werr == nil) {
			t.Fatalf("got (%d,%d,%v) want (%d,%d,%v)", gnd, gns, gerr, wnd, wns, werr)
		}
		if string(gotDst[:gnd]) != string(wantDst[:wnd]) {
			t.Fatalf("data mismatch")
		}
	})
}

func benchData() []byte { return randBytes(2, 1<<20) }

func BenchmarkEncode(b *testing.B) {
	src := benchData()
	dst := make([]byte, MaxEncodedLen(len(src)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(dst, src)
	}
}

func BenchmarkEncodeStdlib(b *testing.B) {
	src := benchData()
	dst := make([]byte, stdascii85.MaxEncodedLen(len(src)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stdascii85.Encode(dst, src)
	}
}
