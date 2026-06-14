//go:build s390x

package ascii85

// decodeGroups decodes blocks*4 groups of 5 base-85 chars into 4 bytes each (20
// chars -> 16 bytes per block) via the z/Architecture vector facility. Generated
// by go-asmgen (decode_s390x_gen.go), in decode_s390x.s. The caller guarantees
// every char is a regular base-85 digit ('!'..'u').
func decodeGroups(dst, src []byte, blocks int)

// decodeSIMD decodes the leading run of clean groups into 4 bytes each, using the
// vector kernel for whole 4-group blocks and a scalar tail for the (groups&3)
// remainder.
func decodeSIMD(dst, src []byte, groups int) {
	blocks := groups / 4
	if blocks > 0 {
		decodeGroups(dst, src, blocks)
	}
	done := blocks * 4
	if rem := groups - done; rem > 0 {
		decodeTail(dst[done*4:], src[done*5:], rem)
	}
}
