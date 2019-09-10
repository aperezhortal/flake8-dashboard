# -*- coding: utf-8 -*-
"""A plugin for flake8 to generate HTML reports."""
import os
from collections import defaultdict
from collections import namedtuple

import pandas
from flake8.formatting import base
from pathlib import PurePath
import itertools

# A sequence of error code prefixes
# The first matching prefix determines the severity

SEVERITY_ORDER = [
    ('E9', 1),
    ('F', 1),
    ('E', 2),
    ('W', 2),
    ('C', 2),
    ('D', 3)
]
DEFAULT_SEVERITY = 3

SEVERITY_NAMES = [
    'high',
    'medium',
    'low'
]


def find_severity(code):
    """Given a flake8-style error code, return an ordinal severity."""
    for prefix, sev in SEVERITY_ORDER:
        if code.startswith(prefix):
            return sev
    return DEFAULT_SEVERITY


IndexEntry = namedtuple(
    'IndexEntry',
    'filename report_name error_count highest_sev'
)


def full_split(_path):
    intermediate_paths = list()
    tail = _path

    while len(tail) > 0:
        head, tail = os.path.split(_path)
        _path = head
        if len(_path) > 0:
            intermediate_paths.append(_path)

    return intermediate_paths


class Reporter(base.BaseFormatter):
    """A plugin for flake8 to render errors as HTML reports."""

    def after_init(self):
        """Configure the plugin run."""

        self.files = []
        self.error_counts = {}
        self.file_count = 0
        self.errors = []
        self.by_code = defaultdict(list)

    def beginning(self, filename):
        """Reset the per-file list of errors."""

        self.file_count += 1

    def handle(self, error):
        """Record this error against the current file."""

        severity = find_severity(error.code)

        error_input = error._asdict()

        error_input['severity'] = severity

        self.errors.append(error_input)

    def finished(self, filename):
        """Write the HTML reports for filename."""
        pass

    def stop(self):
        """After the flake8 run, write the stylesheet and index."""

        # Save in a pandas dataframe all the errors
        error_db = pandas.DataFrame(self.errors,
                                    columns=['filename',
                                             'code',
                                             'line_number',
                                             'text',
                                             'column_number',
                                             'physical_line',
                                             'severity'
                                             ]
                                    )

        error_db.rename(columns={"filename": "path"}, inplace=True)

        file_list = error_db['path'].to_list()

        #########################################
        # Remove the project's root from the path
        common_prefix = os.path.commonprefix(file_list)

        error_db['path'] = error_db['path'].apply(
            lambda _path: os.path.relpath(_path, common_prefix)
        )

        ##########################################
        # Separate the path in parent and filename
        # Save the dataframe to a file

        tmp = error_db['path'].apply(os.path.split)

        error_db['parent'] = tmp.apply(lambda x: x[0])
        error_db['file_name'] = tmp.apply(lambda x: x[1])

        error_db['level'] = error_db['path'].apply(
            lambda _path: len(PurePath(_path).parts)
        )

        error_db.to_pickle("report.pickle")

        #######################################################
        # Compute the aggregated statistics by module/directory

        file_info = error_db[['path']].drop_duplicates(subset=['path'])
        file_info.set_index('path', inplace=True)

        file_info['errors_count'] = error_db["path"].value_counts()

        file_info.reset_index(level=0, inplace=True)

        tmp = file_info['path'].apply(os.path.split)
        file_info['parent'] = tmp.apply(lambda x: x[0])
        # Keep the last directory/module as id
        file_info['id'] = tmp.apply(lambda x: x[1])

        total_errors_by_file = error_db["path"].value_counts()

        aggregated_totals = total_errors_by_file.copy()

        intermediate_paths = file_info["path"].apply(full_split)

        intermediate_paths = tuple(
            itertools.chain.from_iterable(intermediate_paths.values)
        )

        intermediate_paths = set(intermediate_paths)

        for i in intermediate_paths:
            sel = total_errors_by_file.index.get_level_values(0).str.contains(i)
            aggregated_totals.loc[i] = total_errors_by_file[sel].sum()

        aggregated_totals = pandas.DataFrame(data=aggregated_totals.values,
                                             index=aggregated_totals.index,
                                             columns=["errors_count"])

        aggregated_totals['path'] = aggregated_totals.index

        aggregated_totals.reset_index(inplace=True, drop=True)

        aggregated_totals = aggregated_totals[['path', 'errors_count']]

        tmp = aggregated_totals['path'].apply(os.path.split)
        aggregated_totals['parent'] = tmp.apply(lambda x: x[0])
        # Keep the last directory/module as id
        aggregated_totals['id'] = tmp.apply(lambda x: x[1])

        sel = aggregated_totals['parent'] == ""  # Select root path

        total_error_count = aggregated_totals.loc[sel, 'errors_count'].sum()
        aggregated_totals.loc[sel, 'parent'] = "All"
        aggregated_totals.loc[sel, 'id'] = aggregated_totals.loc[sel, 'path']

        aggregated_totals = pandas.concat(
            [pandas.DataFrame.from_dict({"path": ["All"],
                                         "errors_count": [total_error_count],
                                         "parent": [""],
                                         "id": ["All"]}),
             aggregated_totals],
            ignore_index=True,
            sort=False)

        aggregated_totals.to_pickle("statistics.pickle")

        if self.options.sunburst:
            import plotly.graph_objs as go
            import plotly.offline as pyo
            trace = go.Sunburst(
                parents=aggregated_totals.parent.values,
                values=aggregated_totals.errors_count.values,
                ids=aggregated_totals.path.values,
                labels=aggregated_totals.id.values,
                # outsidetextfont = {"size": 20, "color": "#377eb8"},
                branchvalues='total',
                marker={"line": {"width": 2}},
            )

            pyo.iplot([trace])

    @classmethod
    def add_options(cls, options):
        """Add a -- option to the OptionsManager."""
        
        cls.option_manager = options
        options.add_option(
            '--sunburst',
            help="Generate a report using a sunburst plot",
            default=False,
            action="store_true",
            dest="sunburst"
        )
