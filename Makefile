# Build x86_64 and aarch64 s9pks (no riscv — emulated riscv Python builds are
# fragile and StartOS runs the orchestrator only, not on the Sparks).
ARCHES := x86 arm
# overrides to s9pk.mk must precede the include statement
include s9pk.mk
