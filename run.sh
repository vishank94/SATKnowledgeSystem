#!/bin/bash
# Script to run Streamlit app with proper environment variables for macOS

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MKL_NUM_THREADS=1

streamlit run app.py
