# PyTorch MPS Operators Coverage
 
Visit https://qqaatw.github.io/pytorch-mps-ops-coverage/ to check the table.

## Note

The table only contains ops under MPS dispatch key in `native_functions.yaml` for simplicity.

For other dispatch keys and ops outside PyTorch:
1. `CompositeExplicitAutograd`: Most ops under this dispatch key have been supported by MPS.
2. `CompositeImplicitAutograd`: Most ops under this dispatch key have been supported by MPS.
3. `torchvision`: Ops under `torchvision` have not been supported by MPS.

## Contributions

Contributions are welcomed for completing the details of MPS ops coverage in `config.yml`. 

Currently, there are two entries for each op that can be filled:

1. `starting_macOS_version`: The macOS version which starts supporting the op.
2. `note`: A note which describes any worth mentioning details about the op.