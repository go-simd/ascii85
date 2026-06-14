//go:build amd64 || ppc64le || s390x || riscv64 || loong64 || (arm64 && go1.27)

package ascii85

// decodeTail decodes a scalar run of clean 5-char groups (the groups&3 remainder
// that the 4-group-wide SIMD kernel does not cover) from src into dst, both
// already offset to the remainder by the caller. It is the same base-85
// multiply-accumulate as the generic path.
func decodeTail(dst, src []byte, groups int) {
	di := 0
	for g := 0; g < groups; g++ {
		s := src[g*5:]
		v := (((uint32(s[0]-'!')*85+uint32(s[1]-'!'))*85+
			uint32(s[2]-'!'))*85+uint32(s[3]-'!'))*85 + uint32(s[4]-'!')
		d := dst[di : di+4 : di+4]
		d[0] = byte(v >> 24)
		d[1] = byte(v >> 16)
		d[2] = byte(v >> 8)
		d[3] = byte(v)
		di += 4
	}
}
