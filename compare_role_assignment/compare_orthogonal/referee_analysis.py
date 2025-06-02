#!/usr/bin/env python3
"""
Referee-specific clip analysis script.
This script analyzes clips where PRTReid and DBSCAN methods differ specifically for the referee role.
"""

from orthogonal_comparison import analyze_referee_clips

if __name__ == "__main__":
    # Run referee-specific clip analysis
    significant_clips, all_clip_analysis = analyze_referee_clips() 