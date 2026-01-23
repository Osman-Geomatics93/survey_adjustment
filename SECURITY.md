# Security Policy

## Supported Versions

The following versions of Survey Adjustment & Network Analysis are currently supported with security updates:

| Version | Supported          |
|---------|-------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Survey Adjustment & Network Analysis seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT Create a Public Issue

Please **do not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Report Privately

Send a detailed report to:

**Email:** [422436@ogr.ktu.edu.tr](mailto:422436@ogr.ktu.edu.tr)

**Subject Line:** `[SECURITY] Survey Adjustment Plugin Vulnerability Report`

### 3. Include the Following Information

To help us understand and address the issue quickly, please include:

- **Description:** A clear description of the vulnerability
- **Impact:** What could an attacker potentially do?
- **Steps to Reproduce:** Detailed steps to reproduce the issue
- **Affected Versions:** Which version(s) are affected?
- **Potential Fix:** If you have suggestions for how to fix the issue
- **Your Contact Info:** So we can follow up with questions

### Example Report Format

```
VULNERABILITY REPORT
====================

Description:
[Describe the vulnerability]

Impact:
[Describe potential impact]

Steps to Reproduce:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Affected Version(s):
- 1.0.2

Environment:
- QGIS Version: 3.34
- Operating System: Windows 11

Suggested Fix:
[If applicable]
```

## What to Expect

| Timeline | Action |
|----------|--------|
| **24-48 hours** | Acknowledgment of your report |
| **1 week** | Initial assessment and response |
| **2-4 weeks** | Fix development (depending on complexity) |
| **Upon fix** | Credit in release notes (if desired) |

## Security Best Practices for Users

When using the plugin:

1. **Download from Official Sources**
   - GitHub Releases: https://github.com/Osman-Geomatics93/survey_adjustment/releases
   - QGIS Plugin Repository

2. **Keep Updated**
   - Always use the latest version
   - Check for updates regularly in QGIS Plugin Manager

3. **Verify Input Data**
   - Only process CSV files from trusted sources
   - Review data before processing sensitive surveys

4. **Protect Output Files**
   - JSON and HTML reports may contain coordinate data
   - Store outputs securely if data is sensitive

## Scope

This security policy covers:

- The Survey Adjustment & Network Analysis QGIS plugin
- Code in this GitHub repository
- Official releases

This policy does **not** cover:

- Third-party forks or modifications
- QGIS itself (report to QGIS security team)
- User's local environment or network security

## Recognition

We appreciate responsible disclosure. Security researchers who report valid vulnerabilities will be:

- Credited in the release notes (unless anonymity is requested)
- Thanked publicly on the repository (with permission)

---

Thank you for helping keep Survey Adjustment & Network Analysis secure!
