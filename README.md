# Photoreceptor Tracking
A photoreceptor tracking for drosophilia retina

## Installation
You will need to have `git`, `python` and `pip` installed.

1. Clone the repository and enter the folder:
```bash
git clone https://github.com/hugo-paugesteros/tracking.git
cd tracking

```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the environment:
    - Mac / linux
    ```bash
    source venv/bin/activate
    ```
    - Windows
    ```bash
    venv\Scripts\activate
    ```

4. Install the application:
```bash
pip install -e .
```


## How to run
Every time you want to use the tool, double-click one of these in the `tracking` folder:
- **Mac**: `launch_tracker.command`
- **Windows**: `launch_tracker.bat`
- **Linux**: `launch_tracker.sh` (if double-clicking doesn't run it, right-click it and choose "Run" or "Run as a Program")

A terminal window will open and stay open - if anything goes wrong, the error message will be there so you can copy it for a bug report (see below).

Alternatively, you can still open your terminal, activate your environment (Step 3), and type:
```bash
launch-tracker
```

## How to update
From time to time, check this page to see if I updated the code. Open your terminal, navigate to your local `tracking` folder, and simply type:
```bash
git pull
```

## Found a bug? Want a new feature?
In GitHub, you can open a ticket in the "Issues" tab at the top of the page. If you prefer, you can also send me an email directly: hugo [at] paugesteros.com

To help me fix things, please include:
* A brief description of what you were doing when the tool crashed.
* The error message: Copy and paste the entire block of scary-looking text that appeared in your terminal window.
* A screenshot of the Napari interface if the visual display looks wrong.