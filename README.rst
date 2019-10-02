================
flake8-dashboard
================

A flake8 plugin to generate a HTML dashboard with a report of all the flake8 violations.

Installation
============

If flake8 is not installed, run:

.. code-block:: bash

   $ pip install flake8

Finally, to install the latest release of the plugin from the
Python Package Index, run:

.. code-block:: bash

   $ pip install flake8-dashboard

Or to install the latest development version (master branch), run:

.. code-block:: bash

   $ pip install git+https://github.com/aperezhortal/flake8-dashboard

Usage
=====

Run flake8 with the ``--format=dashboard`` option to create a nice-looking dashboard.

Options:

- ``--outputdir=<output_dir>``: Directory to save the HTML output ("./flake8_dashboard" by default).
- ``--debug-info``: Write additional debugging information as csv format (flake8 violations and aggregations).
- ``--title=<title>``: Set the dashboard's title. No title by default.

Simple usage example:

.. code-block:: bash

   $ flake8 --format=dashboard --outputdir=flake-report --title="My dashboard"


Demo
~~~~

`Check a demo here! <https://aperezhortal.github.io/flake8-dashboard/example_dashboard/index.html>`_


Credits
=======

- This package was created using the `flake8-html`_ as a template.

- The dashboard html page was created using the
  `light-bootstrap-dashboard`_ template by `Creative Tim`_.

- The interactive plots are created using `Plotly Python`_ .

.. _light-bootstrap-dashboard: https://demos.creative-tim.com/light-bootstrap-dashboard/
.. _`Creative Tim`: https://www.creative-tim.com/
.. _`Plotly Python`: https://plot.ly/python/
.. _flake8-html: https://github.com/lordmauve/flake8-html




