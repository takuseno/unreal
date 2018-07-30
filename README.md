# unreal
UNREAL implementation with TensorFlow

https://arxiv.org/abs/1611.05397

Parallel execution is done by Distributed TensorFlow as multithreading is extremely slow with auxiliary tasks because of GIL.

## dependencies
- Python3
- tensorflow
- opencv-python
- git+https://github.com/imai-laboratory/rlsaber

## train
```sh
$ ./run.sh [-e environment_id] [-r (render)] [-n num_of_processes]
```

## ToDo
- [x] Add last actions and immediate rewards to input
- [x] Implement reward prediction
- [x] Implement value function replay
- [ ] Implement pixel control
- [ ] Implement input change prediction

## implementations
Base A3C implementation is based on https://github.com/takuseno/a3c .
