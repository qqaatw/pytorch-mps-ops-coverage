# PyTorch MPS Operators Coverage
 
Visit https://qqaatw.github.io/pytorch-mps-ops-coverage/ to check the table.

The table is updated automatically every day.

## Note

The table only contains ops under MPS dispatch key in `native_functions.yaml` for simplicity. Manually registered ops must be updated in [config.yml](config.yml).

For other dispatch keys and ops outside PyTorch:
1. `CompositeExplicitAutograd`: Most ops under this dispatch key have been supported by MPS.
2. `CompositeImplicitAutograd`: Most ops under this dispatch key have been supported by MPS.
3. `torchvision`: Except for `deform_conv2d`, other ops such as `nms` have been supported by MPS.

## Contributions

Contributions are welcomed for completing the details of MPS ops coverage in [config.yml](config.yml).

Currently, there are three entries for each op that can be filled:

1. `starting_macOS_version`: The macOS version which starts supporting the op.
2. `note`: A note which describes any worth mentioning details about the op.
3. `included_in_latest`: For manually registered ops, use this to indicate whether the ops are included in the latest PyTorch release.