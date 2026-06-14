package ascii85

import "encoding/ascii85"

// Decode decodes src into dst, returning the number of bytes written to dst and
// consumed from src.
//
// Decoding is byte-, error-, and offset-identical to encoding/ascii85.Decode. A
// SIMD kernel handles the common case — a leading run of "clean" 5-char groups,
// i.e. groups whose bytes are all in the regular base-85 range ['!','u']
// (33..117) — turning each group into 4 bytes with a base-85 multiply-accumulate
// (the inverse of the encoder's reciprocal-divide). Anything the kernel must not
// guess about — whitespace, the all-zero "z" shortcut, the short/invalid
// trailing group, CorruptInputError offsets, and flush handling — is delegated
// to encoding/ascii85.Decode on the remaining input, so the observable behaviour
// matches the standard library exactly.
//
// The decoder handles 4-byte chunks, using a special encoding for the last
// fragment, so Decode is not appropriate for use on individual blocks of a large
// data stream. Use encoding/ascii85.NewDecoder for streaming.
func Decode(dst, src []byte, flush bool) (ndst, nsrc int, err error) {
	// Count the maximal leading run of whole 5-char groups whose every byte is a
	// regular base-85 digit ('!'..'u'). Such a group is unambiguous: the stdlib
	// decoder would consume exactly 5 src bytes and emit exactly 4 dst bytes for
	// it, with v and nb resetting to 0 at the boundary — a clean resume point.
	// Stop early if dst lacks room for another 4-byte group (mirroring stdlib's
	// "len(dst)-ndst < 4" early return), or at the first byte that is not a
	// regular digit (whitespace, 'z', or out of range), handing the rest to the
	// stdlib so every edge case stays identical.
	maxGroups := len(src) / 5
	if g := len(dst) / 4; g < maxGroups {
		maxGroups = g
	}
	groups := 0
	for groups < maxGroups {
		off := groups * 5
		if !clean5(src[off : off+5]) {
			break
		}
		groups++
	}
	if groups > 0 {
		decodeSIMD(dst, src, groups)
		ndst = groups * 4
		nsrc = groups * 5
	}
	// Delegate the remainder; offsets returned by the stdlib are relative to the
	// remaining slices, so shift them (and any CorruptInputError) back.
	rnd, rns, rerr := ascii85.Decode(dst[ndst:], src[nsrc:], flush)
	if rerr != nil {
		// The stdlib discards partial progress on error (it returns 0, 0,
		// CorruptInputError), so match that exactly; only the offset shifts.
		if ce, ok := rerr.(CorruptInputError); ok {
			rerr = CorruptInputError(int(ce) + nsrc)
		}
		return 0, 0, rerr
	}
	ndst += rnd
	nsrc += rns
	return ndst, nsrc, nil
}

// clean5 reports whether all five bytes are regular base-85 digits ('!'..'u',
// i.e. 33..117) — the only groups the SIMD kernel may decode. 'z', whitespace,
// and any out-of-range byte fail, so they fall back to the stdlib.
func clean5(g []byte) bool {
	return g[0] >= '!' && g[0] <= 'u' &&
		g[1] >= '!' && g[1] <= 'u' &&
		g[2] >= '!' && g[2] <= 'u' &&
		g[3] >= '!' && g[3] <= 'u' &&
		g[4] >= '!' && g[4] <= 'u'
}
