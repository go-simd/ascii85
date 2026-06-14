//go:build !amd64 && !ppc64le && !s390x && !riscv64 && !loong64 && !(arm64 && go1.27)

package ascii85

// encodeSIMD has no SIMD kernel on this arch (Go's assembler exposes no vector
// integer multiply for it, so the base-85 division cannot be vectorised). It
// encodes the leading run of groups with a tight scalar loop instead. The caller
// guarantees src[:groups*4] contains no all-zero group, so each group always
// expands to exactly 5 chars and the all-zero "z" branch is skipped entirely,
// making this marginally faster than encoding/ascii85's general loop while
// remaining byte-identical.
func encodeSIMD(dst, src []byte, groups int) int {
	di := 0
	for g := 0; g < groups; g++ {
		s := src[g*4:]
		v := uint32(s[0])<<24 | uint32(s[1])<<16 | uint32(s[2])<<8 | uint32(s[3])
		// Five base-85 digits, most significant first.
		d := dst[di : di+5 : di+5]
		d[4] = '!' + byte(v%85)
		v /= 85
		d[3] = '!' + byte(v%85)
		v /= 85
		d[2] = '!' + byte(v%85)
		v /= 85
		d[1] = '!' + byte(v%85)
		v /= 85
		d[0] = '!' + byte(v)
		di += 5
	}
	return di
}
