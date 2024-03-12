#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def test_help_option(script_runner):
    ret = script_runner.run('dwiml_compute_connectivity_matrix_from_labels.py',
                            '--help')
    assert ret.success


def test_execution(script_runner):
    # Impossible for now, no labels file. Could use data from scilpy's tests.
    pass
