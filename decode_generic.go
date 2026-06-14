//go:build !amd64 && !ppc64le && !s390x && !riscv64 && !loong64

package ascii85

// decodeSIMD has no SIMD kernel on this arch (Go's assembler exposes no vector
// integer multiply for it, so the base-85 multiply-accumulate cannot be
// vectorised). It decodes the leading run of clean 5-char groups with a tight
// scalar loop instead. The caller guarantees src[:groups*5] is groups whole
// groups of regular base-85 digits ('!'..'u') and that dst has room for
// groups*4 bytes, so each group always yields exactly 4 bytes.
func decodeSIMD(dst, src []byte, groups int) {
	di := 0
	for g := 0; g < groups; g++ {
		s := src[g*5:]
		// Base-85 multiply-accumulate: the inverse of the encoder's reciprocal
		// divide. Each char is a digit in [0,84] after subtracting '!'.
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
