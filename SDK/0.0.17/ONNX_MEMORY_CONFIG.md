# ONNX memory settings

Set these top-level JSON booleans in `sdk_model.json` (ASR) or `sdk_defaults.json` (WUW):

```json
{
  "enable_weight_prepacking": false,
  "use_device_allocator_for_initializers": false
}
```

Missing keys use the values above. Values must be JSON booleans. Restart the engine after editing. Each engine applies its own settings before creating all its ONNX sessions.

Disabling prepacking avoids optimized copies of weights and can reduce RSS at the cost of inference speed. Enabling the device allocator setting allocates initializer weights outside the CPU arena; it does not select a GPU. Its memory and timing effect depends on the model and ONNX Runtime version.

4wheels disables prepacking; POC explicitly enables it. Both leave device allocation disabled. These settings require no rebuild and have no C/C++ API override.
