# ai.edu

A common repository for AI/ML learning resources, organized as a collection of Git submodules.

## Subrepositories

| Name | Source | Description |
|------|--------|-------------|
| [pytorch](pytorch/) | [pytorch/examples](https://github.com/pytorch/examples) | A set of examples around PyTorch in Vision, Text, Reinforcement Learning, etc. |

## Getting Started

Clone the repository with all submodules:

```bash
git clone --recurse-submodules https://github.com/audrus1917/ai.edu.git
```

Or, if you have already cloned the repository, initialize and update the submodules:

```bash
git submodule update --init --recursive
```

## Updating Submodules

To pull the latest changes from all submodules:

```bash
git submodule update --remote --merge
```