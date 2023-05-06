import argparse
import json
import os
import re
import sys

from hypernets.tabular import get_tool_box
from hypernets.utils import const


def _stub():
    pass


def _parse_pairs(pairs):
    result = {}

    if pairs is not None:
        for p in pairs:
            i = p.find('=')
            assert i > 0, f''
            key = p[:i].strip()
            value = p[i + 1:].strip()
            if re.match(r'[+-]?[0-9]+', value):
                value = int(value)
            elif re.match(r'[+-]?[0-9]+\.[0-9]+', value):
                value = float(value)
            result[key] = value
    return result


def detect_estimator(name_or_cls, task, *,
                     toolbox='default',
                     init_kwargs=None, fit_kwargs=None, n_samples=100, n_features=5):
    tb = get_tool_box(toolbox)
    return tb.estimator_detector(name_or_cls, task,
                                 init_kwargs=init_kwargs,
                                 fit_kwargs=fit_kwargs,
                                 n_samples=n_samples,
                                 n_features=n_features)()


def detect_with_process(name_or_cls, task, *,
                        toolbox='default', init_kwargs=None, fit_kwargs=None, n_samples=100, n_features=5):
    from hypernets.dispatchers.process import LocalProcess
    from hypernets.utils import is_os_windows
    import tempfile

    my_mod = _stub.__module__
    cmd = f'{sys.executable} -m {my_mod} --estimator {name_or_cls} --toolbox={toolbox}' \
          f' --task {task} --samples {n_samples} --features {n_features}'
    if init_kwargs:
        cmd += ' --init'
        for k, v in init_kwargs.items():
            cmd += f' {k}={v}'
    if fit_kwargs:
        cmd += ' --fit'
        for k, v in fit_kwargs.items():
            cmd += f' {k}={v}'

    out_file = tempfile.mktemp(prefix='detect_estimator', suffix='out')
    err_file = tempfile.mktemp(prefix='detect_estimator', suffix='err')
    proc = LocalProcess(cmd, in_file='', out_file=out_file, err_file=err_file)
    proc.run()
    exitcode = proc.exitcode

    if is_os_windows:
        assert exitcode is None or exitcode == 0, f'Failed to run cmd: {cmd}'
    else:
        assert exitcode == 0, f'Failed to run cmd: {cmd}'

    with open(out_file, 'r', encoding="UTF-8") as f:
        data = f.read()
        index = data.find("]")
        result = json.loads(data[: index+1])

    try:
        os.remove(out_file)
        os.remove(err_file)
    except:
        pass

    return result


def main():
    a = argparse.ArgumentParser(description='Command line utility to detect estimator state')
    a.add_argument('--estimator', type=str, default='lightgbm.LGBMClassifier',
                   help='estimator class name, default %(default)s"')

    a.add_argument('--task', type=str, default=const.TASK_BINARY,
                   choices=[const.TASK_BINARY, const.TASK_MULTICLASS, const.TASK_REGRESSION],
                   help='the number of samples, default %(default)s')
    a.add_argument('--toolbox', type=str, default='default',
                   choices=['default', 'pandas', 'cuml', 'dask'],
                   help='the toolbox alias, default %(default)s')
    a.add_argument('--samples', '-N', type=int, default=100, metavar='N',
                   help='the number of samples, default %(default)s')
    a.add_argument('--features', '-F', type=int, default=5, metavar='F',
                   help='the number of features, default %(default)s')

    a.add_argument('--init', '-I', type=str, nargs='*', metavar='INIT_KV',
                   help='estimator init option pairs in the form of "key=value"')
    a.add_argument('--fit', '-T', type=str, nargs='*', metavar='FIT_KV',
                   help='estimator fit option pairs in the form of "key=value"')

    args = a.parse_args()
    init_kwargs = _parse_pairs(args.init)
    fit_kwargs = _parse_pairs(args.fit)

    result = detect_estimator(args.estimator, args.task,
                              toolbox=args.toolbox,
                              init_kwargs=init_kwargs,
                              fit_kwargs=fit_kwargs,
                              n_samples=args.samples,
                              n_features=args.features)
    print(json.dumps(list(result)))


if __name__ == '__main__':
    from hypernets.utils import logging

    logging.set_level('error')
    main()
