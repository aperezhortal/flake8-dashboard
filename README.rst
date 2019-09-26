================
flake8-dashboard
================

A flake8 plugin to generate a HTML dashboard with a report of all the flake8 violations.

Installation
~~~~~~~~~~~~
If flake8 is not installed, run:

.. code-block:: bash

   $ pip install flake8

If anaconda is used:

.. code-block:: bash

   $ conda install flake8

Finally, to install the plugin run:

.. code-block:: bash

   $ pip install git+https://github.com/aperezhortal/flake8-dashboard

Usage
~~~~~

Run flake8 with the ``--format=dashboard`` option to create a nice-looking
dashboard report in HTML.
The directory in which to write HTML output can be specified with the ``--outputdir``:
(by default is "./flake8_dashboard").
Additionally, debuggin information (flake8 violations and aggregations) can
be saved as csv format passing the ``--debug-info`` option.

Simple usage example:

.. code-block:: bash

   $ flake8 --format=dashboard --outputdir=flake-report


Demo
~~~~

`Check a demo here! <example_dashboard/index.html>`_


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




