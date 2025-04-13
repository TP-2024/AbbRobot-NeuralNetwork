#!/bin/bash

# Detect the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS configuration
    source venv/bin/activate
    source .venv/bin/activate
    export QT_PLUGIN_PATH=/Applications/Webots.app/Contents/MacOS/plugins
    export WEBOTS_HOME=/Applications/Webots.app/Contents
    if [[ "$1" == "train" ]]; then
      /Applications/Webots.app/Contents/MacOS/webots --batch --minimize --no-rendering --mode=fast --stdout --stderr --extern-urls ./dqn/worlds/inverse_kinematics.wbt
    else
      /Applications/Webots.app/Contents/MacOS/webots ./dqn/worlds/inverse_kinematics.wbt
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]; then
    # Windows configuration
    source venv/Scripts/activate
    export WEBOTS_HOME="C:/Program Files/Webots"
    export QT_PLUGIN_PATH="C:/Program Files/Webots/msys64/mingw64/plugins"
    "C:/Program Files/Webots/msys64/mingw64/bin/webots.exe"
else
    echo "Unsupported operating system"
    exit 1
fi
