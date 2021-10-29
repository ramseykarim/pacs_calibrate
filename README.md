# pacs_calibrate
Zero-point calibration for Herschel PACS Photometric observations based on the Planck GNILC thermal dust model.

This code has been around in my [HELPSS repository](https://github.com/ramseykarim/helpss) for a couple years. The first push of the "modern" code is in this [July 26, 2019 commit](https://github.com/ramseykarim/helpss/commit/40f5e51243f3178950b693336488799d98d54a06) to that repo.
I have been working with this calibration since early 2018 when I started working with Lee Mundy (with whom I also developed [Alpha-X](https://github.com/ramseykarim/alpha-x)) and Tracy Huard on the HELPSS project to process Herschel photometry and generate temperature and column density maps. See [mantipython](https://github.com/ramseykarim/mantipython) for code for that.

The purpose of the code is to apply the necessary zero-point correction to the Herschel PACS phometric observations. This correction has already been applied to SPIRE photometry available on the archive, but is not applied to archival PACS photometry. This program is designed to run on Level 2 or 2.5 PACS photometry data products. It assumes you also have SPIRE photometry available (which is used to do some masking), but this isn't strictly necessary, so one could remove these steps from the process.

Feel free to adapt this code to your own needs or contact me for help!
