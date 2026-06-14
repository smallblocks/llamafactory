import { sdk } from '../sdk'
import { configFile } from '../file-models/config'

const { InputSpec, Value } = sdk

const inputSpec = InputSpec.of({
  model: Value.text({
    name: 'Base Model',
    description: 'Hugging Face model ID to fine-tune.',
    required: true,
    default: 'google/gemma-4-31B-it',
  }),
  template: Value.text({
    name: 'Chat Template',
    description:
      'LLaMA-Factory chat template. Use "gemma4" if your LLaMA-Factory build ' +
      'ships it, otherwise "gemma".',
    required: true,
    default: 'gemma',
  }),
  finetuningType: Value.select({
    name: 'Fine-tuning Type',
    description: 'LoRA is recommended and fits a 31B on one Spark. Full needs both.',
    default: 'lora',
    values: { lora: 'LoRA (recommended)', full: 'Full fine-tune' },
  }),
  quantization: Value.select({
    name: 'Quantization',
    description:
      'bf16 is recommended on a Spark (128GB unified memory fits a 31B LoRA). ' +
      'Choose 4-bit QLoRA only to save memory at some quality cost.',
    default: 'none',
    values: { none: 'bf16 (no quantization)', '4': '4-bit QLoRA' },
  }),
  loraRank: Value.number({
    name: 'LoRA Rank',
    description: 'Adapter rank. 8–32 is typical for small datasets.',
    required: true,
    default: 16,
    integer: true,
    min: 1,
    max: 256,
  }),
  loraAlpha: Value.number({
    name: 'LoRA Alpha',
    description: 'Usually 2× the rank.',
    required: true,
    default: 32,
    integer: true,
    min: 1,
  }),
  learningRate: Value.text({
    name: 'Learning Rate',
    description: 'e.g. 1.0e-4 for LoRA.',
    required: true,
    default: '1.0e-4',
  }),
  epochs: Value.number({
    name: 'Epochs',
    description:
      'Passes over the dataset. With a tiny dataset, keep this low (1–3) to ' +
      'avoid overfitting.',
    required: true,
    default: 3,
    integer: false,
    min: 0.1,
  }),
  cutoffLen: Value.number({
    name: 'Cutoff Length (tokens)',
    description: 'Max sequence length. Longer = more memory.',
    required: true,
    default: 4096,
    integer: true,
    min: 128,
  }),
  perDeviceBatchSize: Value.number({
    name: 'Per-device Batch Size',
    description: 'Sequences per GPU step.',
    required: true,
    default: 1,
    integer: true,
    min: 1,
  }),
  gradAccumSteps: Value.number({
    name: 'Gradient Accumulation Steps',
    description: 'Effective batch = batch size × accumulation × #GPUs.',
    required: true,
    default: 8,
    integer: true,
    min: 1,
  }),
})

export const trainingDefaults = sdk.Action.withInput(
  'training-defaults',

  async ({ effects }) => ({
    name: 'Training Defaults',
    description: 'Set the model and default hyperparameters for fine-tunes.',
    warning: null,
    allowedStatuses: 'any',
    group: null,
    visibility: 'enabled',
  }),

  inputSpec,

  async ({ effects }) => {
    const cfg = await configFile.read().const(effects)
    if (!cfg) return {}
    return {
      model: cfg.model,
      template: cfg.template,
      finetuningType: cfg.finetuningType,
      quantization: (cfg.quantizationBit === 4 ? '4' : 'none') as 'none' | '4',
      loraRank: cfg.loraRank,
      loraAlpha: cfg.loraAlpha,
      learningRate: cfg.learningRate,
      epochs: cfg.epochs,
      cutoffLen: cfg.cutoffLen,
      perDeviceBatchSize: cfg.perDeviceBatchSize,
      gradAccumSteps: cfg.gradAccumSteps,
    }
  },

  async ({ effects, input }) => {
    await configFile.merge(effects, {
      model: input.model,
      template: input.template,
      finetuningType: input.finetuningType,
      quantizationBit: input.quantization === '4' ? 4 : null,
      loraRank: input.loraRank,
      loraAlpha: input.loraAlpha,
      learningRate: input.learningRate,
      epochs: input.epochs,
      cutoffLen: input.cutoffLen,
      perDeviceBatchSize: input.perDeviceBatchSize,
      gradAccumSteps: input.gradAccumSteps,
    })
    return {
      version: '1',
      title: 'Defaults Saved',
      message: 'These values pre-fill the Web UI when you start a new run.',
      result: { type: 'single', value: input.model, copyable: true, qr: false, masked: false },
    }
  },
)
