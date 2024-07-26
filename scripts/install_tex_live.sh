#!/bin/zsh

# Set installation directory
TEXLIVE_DIR=$HOME/texlive
INSTALLER_DIR=$HOME/texlive-installer

# Download TeX Live installer
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz

# Create installer directory and extract the installer
mkdir -p $INSTALLER_DIR
tar -xzf install-tl-unx.tar.gz -C $INSTALLER_DIR --strip-components=1

# Change to the installer directory
cd $INSTALLER_DIR

# Run the installer with guided input
echo "Please follow these steps carefully:"
echo "1. When prompted, enter 'D' to set directories."
echo "2. When asked for TEXDIR, enter: $TEXLIVE_DIR"
echo "3. Press Enter to accept the default values for other directories."
echo "4. Enter 'R' to return to the main menu."
echo "5. Finally, enter 'I' to start the installation."
echo "Press Enter to continue..."
read

# Start the installer
./install-tl

# Check if installation was successful
if [ ! -d "$TEXLIVE_DIR/bin" ]; then
    echo "Installation failed. Please run the script again and follow the instructions carefully."
    exit 1
fi

# Add TeX Live to PATH in .zshrc if not already present
if ! grep -q "$TEXLIVE_DIR/bin" $HOME/.zshrc; then
    echo "export PATH=$TEXLIVE_DIR/bin/x86_64-linux:\$PATH" >> $HOME/.zshrc
    echo "Added TeX Live to PATH in .zshrc"
fi

# Source .zshrc to update PATH
source $HOME/.zshrc

# Install XeLaTeX
echo "Installing XeLaTeX..."
$TEXLIVE_DIR/bin/x86_64-linux/tlmgr install xetex

# Verify installation
if command -v xelatex &> /dev/null; then
    echo "XeLaTeX installation successful."
    xelatex --version
else
    echo "XeLaTeX installation failed. Please check your installation and try again."
    exit 1
fi

# Clean up
rm -f $HOME/install-tl-unx.tar.gz
rm -rf $INSTALLER_DIR

echo "Installation complete. Your PATH has been updated. If XeLaTeX is not recognized in new terminals, please restart your terminal or run 'source ~/.zshrc'."