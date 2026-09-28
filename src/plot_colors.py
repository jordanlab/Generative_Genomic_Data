"""
Shared color registry for real/synthetic sample plotting.

Every script that overlays "real" population samples against synthetic
samples (conditional_wgan_pipeline.py, pca_diagnostic.py, pca.py) should
pull colors from here instead of hardcoding hex values locally. This
guarantees SYNTHETIC_COLOR never collides with a population's real color,
even as new populations are added to SAMPLE_COLORS.
"""

# Real-sample color per population, by convention.
SAMPLE_COLORS = {
    'BAQ':  '#E63946',  # red
    'COR':  '#2A9D8F',  # teal
    'VDC2': '#F4A261',  # sandy orange
    'ATQ3': '#7B2FBE',  # purple
    # Added for the 2026-09 sweep. Hues chosen to sit clear of the four
    # above, of SYNTHETIC_COLOR, and of the AFR/EUR/AMR reference colors,
    # so figures stay tellable apart when compared side by side.
    'BOG':  '#117733',  # forest green
    'SAI':  '#882255',  # wine
    'UKB':  '#8C510A',  # brown
    'VDC3': '#999933',  # olive
}
DEFAULT_REAL_COLOR = '#E63946'

# Reserved for synthetic samples in every overlay plot. Must stay distinct
# from every value in SAMPLE_COLORS (and ideally from AFR/EUR/AMR reference
# colors: '#2166AC', '#E87D00', '#89CFF0').
SYNTHETIC_COLOR = '#FFB703'  # amber


def get_real_color(sample_name: str) -> str:
    """Real-sample plot color for a population, falling back to the default."""
    return SAMPLE_COLORS.get(sample_name, DEFAULT_REAL_COLOR)
