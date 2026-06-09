# Installation

## Requirements

- **QGIS 3.22** or later
- **NumPy** (ships with QGIS)
- No other dependencies — the statistics are implemented without SciPy.

## In QGIS

### From the Plugin Manager (recommended)

1. Open QGIS.
2. Go to **Plugins → Manage and Install Plugins**.
3. Search for *Survey Adjustment*.
4. Click **Install Plugin**.

### From a ZIP

1. Download the latest release from the
   [GitHub Releases](https://github.com/Osman-Geomatics93/survey_adjustment/releases) page.
2. In QGIS, go to **Plugins → Manage and Install Plugins → Install from ZIP**.
3. Select the downloaded ZIP and click **Install Plugin**.

Once installed, the tools appear in the **Processing Toolbox** under
*Survey Adjustment*.

### Manual install (from source)

```bash
git clone https://github.com/Osman-Geomatics93/survey_adjustment.git
```

Copy the folder into your QGIS plugins directory:

=== "Windows"

    ```bat
    xcopy /E /I survey_adjustment "%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\survey_adjustment"
    ```

=== "Linux / macOS"

    ```bash
    cp -r survey_adjustment ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
    ```

Then enable the plugin in the Plugin Manager.

## As a standalone Python library

The numerical core (`core/`) is QGIS-independent and depends only on NumPy, so
it can be imported and scripted outside QGIS — useful for automated pipelines,
testing, and reproducible research.

```python
from survey_adjustment.core.models import Network, Point
from survey_adjustment.core.solver import adjust_network_2d
```

!!! tip "Running the tests"
    The test suite runs without QGIS:

    ```bash
    pip install numpy pytest
    pytest -v
    ```
