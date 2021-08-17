## Prerequisites

1. Download [polyvore image data](https://github.com/xthan/polyvore-dataset/) and put ``test_images/`` into ``data`` directory.
2. Download [pretrained model](https://drive.google.com/file/d/1WAErKHDmDfamZQt90wAOC5Db04euIeIP/view?usp=sharing) and put it into ``mcn`` directory
3. Install dependencies by ``pip install dash dash-bootstrap-component``
4. Run ``python main.py`` in ``app`` directory

## Guide

The definition of each arguments:

- **Top**, **Bottom**, **Shoe**, **Bag**, **Accessory**, select an item in an outfit by choosing ID or upload local images. The preview will be shown on the right **Current outfit** pane instantly.
- **The most time to try for each item**: The more this value is, the more time will be spent to get outcome.
- **Sumbit**: submit current outfit for diagnosis and automatic revision."
