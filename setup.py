from setuptools import setup


setup(
    name='qiskit_pulse_simulator',
    version='0.1.0',
    packages=[
        'qiskit_pulse_simulator',
        'qiskit_pulse_simulator.h_str',
    ],
    python_requires='>=3.7',
    install_requires=[
        'qiskit',
        'numpy',
        'qutip',
        'sympy',
    ],
)
