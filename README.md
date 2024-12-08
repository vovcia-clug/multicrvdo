# MultiCRVDO

MultiCRVDO is a Pytorch module that simulates dynamics of multiple CRVDO oscillators using Runge-Kutta 4(5) integration. This model is useful for studying CRVDO.

## Features

- **Simulation of Multiple Oscillators**: Run simulations for multiple instances of coupled oscillators.
- **Customizable Parameters**: Easily alter the parameters for each oscillator and control functions.

## Installation

### Prerequisites

- PyTorch

### Install dependencies

Ensure you have the necessary Pytorch:

https://pytorch.org/get-started/locally/

## Usage

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/vovcia-clug/multicrvdo
cd multicrvdo
```

To run the simulation and visualize the results:

```bash
python multicrvdo.py
```

## Code Structure

- `multicrvdo.py`: Main script containing the CRVDO model definition, simulation, and plotting.

## Example

An example plot will visualize the states of multiple oscillators over time, showing how their states evolve under the given parameters.

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is public domain

## Contact

For any questions or suggestions, please email [vovcia@gmail.com](mailto:vovcia@gmail.com).