# Build x86_64 and aarch64 s9pks (no riscv — emulated riscv Python builds are
# fragile and StartOS runs the orchestrator only, not on the Sparks).
ARCHES := x86 arm
# overrides to s9pk.mk must precede the include statement
include s9pk.mk

# Ensure the JS bundle + assets directory are built BEFORE s9pk.mk's
# list-ingredients/pack step, which needs to load the compiled manifest.
# s9pk.mk cannot resolve INGREDIENTS until javascript/index.js exists
# (chicken-and-egg), so we add an explicit order dependency here.
llama-factory_%.s9pk: javascript/index.js assets
assets:
	@mkdir -p $@
