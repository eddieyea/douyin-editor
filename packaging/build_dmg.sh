#!/bin/bash
# Build the 抖音 Editor macOS .app + DMG for distribution.
# Run this from your Mac: bash packaging/build_dmg.sh
set -e

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$SCRIPT_DIR/build"
APP_NAME="抖音 Editor"
DMG_OUT="$SCRIPT_DIR/DoyinEditor-1.0.dmg"

echo "=== 抖音 Editor — DMG Builder ==="
echo ""

echo "ℹ  Modal credentials are read from each employee's ~/.modal.toml at runtime."
echo "   Run 'modal token set' on each Mac once to enable AI restore."
echo ""

# ─── clean build dir ─────────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ─── create .app bundle ───────────────────────────────────────────────────────
APP="$BUILD_DIR/$APP_NAME.app"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources/douyin-editor"

echo "Building .app bundle..."

# Info.plist
cat > "$APP/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>DoyinEditor</string>
    <key>CFBundleIdentifier</key><string>com.topnotch.douyin-editor</string>
    <key>CFBundleName</key><string>抖音 Editor</string>
    <key>CFBundleDisplayName</key><string>抖音 Editor</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
</dict>
</plist>
PLIST

cp "$SCRIPT_DIR/app_launcher" "$APP/Contents/MacOS/DoyinEditor"
chmod +x "$APP/Contents/MacOS/DoyinEditor"

# Copy project source — exclude things not needed at runtime
rsync -a \
    --exclude='.git' \
    --exclude='.gitignore' \
    --exclude='venv' \
    --exclude='output' \
    --exclude='tmp' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='packaging' \
    --exclude='*.docx' \
    --exclude='make_docs.py' \
    --exclude='calibration' \
    --exclude='assets/gfpgan' \
    --exclude='bin' \
    --exclude='gfpgan' \
    --exclude='README.md' \
    "$PROJECT_ROOT/" "$APP/Contents/Resources/douyin-editor/"

APP_SIZE=$(du -sh "$APP" | cut -f1)
echo "✓ App bundle: $APP_SIZE"
echo ""

# ─── create DMG ──────────────────────────────────────────────────────────────
echo "Creating DMG (this takes ~30 seconds)..."
rm -f "$DMG_OUT"
hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$BUILD_DIR" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_OUT"

DMG_SIZE=$(du -sh "$DMG_OUT" | cut -f1)
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ✓ Done!  DoyinEditor-1.0.dmg  ($DMG_SIZE)            "
echo "╠════════════════════════════════════════════════════════╣"
echo "║  How employees install:                                ║"
echo "║  1. Open DoyinEditor-1.0.dmg                          ║"
echo "║  2. Drag '抖音 Editor' to Applications                 ║"
echo "║  3. Double-click to open (first run: right-click →     ║"
echo "║     Open to bypass Gatekeeper, then click Open)        ║"
echo "║  4. First launch takes ~5–10 min to install packages   ║"
echo "║  5. Browser opens to http://localhost:8765 — done!     ║"
echo "╚════════════════════════════════════════════════════════╝"
