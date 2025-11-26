# Game Cards Styling Reference

## Quick Visual Guide to Enhanced Game Cards

### Card Border & Tile Effect

**Base Card Style:**
```css
.game-card {
    padding: 12px 16px;              /* Compact padding */
    background: var(--secondary-background-color);
    border: 2px solid rgba(128, 128, 128, 0.5);  /* Visible on all 4 sides */
    border-radius: 12px;              /* Rounded corners */
    box-shadow:
        0 4px 8px rgba(0, 0, 0, 0.2),           /* Depth shadow */
        inset 0 1px 0 rgba(255, 255, 255, 0.1); /* Top highlight */
}
```

**Hover Effect:**
```css
.game-card:hover {
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);  /* Deeper shadow */
    transform: translateY(-2px);                 /* Slight lift */
}
```

### Confidence-Based Card Highlighting

#### 🟢 High Confidence (70%+ Win Probability)
```css
.predicted-winner-high {
    border: 3px solid #00ff00;
    background: linear-gradient(135deg,
        rgba(0, 255, 0, 0.12) 0%,
        rgba(0, 255, 0, 0.05) 100%);
    box-shadow:
        0 0 25px rgba(0, 255, 0, 0.6),           /* Green glow */
        0 4px 12px rgba(0, 0, 0, 0.3),           /* Depth */
        inset 0 1px 0 rgba(0, 255, 0, 0.2);      /* Inner highlight */
    animation: pulse-green 2s infinite;
}
```

**Visual Effect:**
- Bright green glowing border
- Subtle green gradient background
- Pulsing glow animation
- Stands out dramatically from other cards

#### 🟡 Medium Confidence (55-69% Win Probability)
```css
.predicted-winner-medium {
    border: 3px solid #ffd700;  /* Gold, not plain yellow */
    background: linear-gradient(135deg,
        rgba(255, 215, 0, 0.1) 0%,
        rgba(255, 215, 0, 0.04) 100%);
    box-shadow:
        0 0 20px rgba(255, 215, 0, 0.5),         /* Golden glow */
        0 4px 12px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 215, 0, 0.15);
    animation: pulse-yellow 2s infinite;
}
```

**Visual Effect:**
- Warm golden border
- Subtle gold gradient background
- Gentle pulsing glow
- Indicates good opportunity without being overwhelming

#### ⚪ Low Confidence (<55% Win Probability)
```css
.predicted-winner-low {
    border: 2px solid rgba(150, 150, 150, 0.6);
    background: var(--secondary-background-color);
    box-shadow:
        0 4px 8px rgba(0, 0, 0, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
```

**Visual Effect:**
- Standard gray border (thinner than high/medium)
- Normal background
- No glow or animation
- Subtle and unobtrusive

### Team Logo Highlighting

When a team is predicted to win, their logo gets highlighted:

#### 🟢 High Confidence Logo
```css
.team-logo-high-confidence {
    padding: 12px;
    background: linear-gradient(135deg,
        rgba(0, 255, 0, 0.2) 0%,
        rgba(0, 255, 0, 0.1) 100%);
    border: 3px solid #00ff00;
    border-radius: 10px;
    box-shadow:
        0 0 20px rgba(0, 255, 0, 0.7),           /* Strong glow */
        inset 0 2px 4px rgba(0, 255, 0, 0.2);    /* Inner depth */
    animation: pulse-glow-green 2s infinite;
}
```

**Visual Effect:**
- Green gradient background behind logo
- Thick green border
- Pulsing green glow
- Makes predicted winner obvious at a glance

#### 🟡 Medium Confidence Logo
```css
.team-logo-medium-confidence {
    padding: 12px;
    background: linear-gradient(135deg,
        rgba(255, 215, 0, 0.15) 0%,
        rgba(255, 215, 0, 0.08) 100%);
    border: 3px solid #ffd700;
    border-radius: 10px;
    box-shadow:
        0 0 15px rgba(255, 215, 0, 0.6),
        inset 0 2px 4px rgba(255, 215, 0, 0.15);
    animation: pulse-glow-yellow 2s infinite;
}
```

### Confidence Badge

New prominent badge shown above prediction details:

#### 🟢 HIGH CONFIDENCE
```css
.confidence-high {
    background: linear-gradient(135deg, #00ff00 0%, #00cc00 100%);
    color: #000;                    /* Black text for maximum contrast */
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.6);
    animation: pulse-badge-green 2s infinite;
}
```

**HTML Output:**
```html
<div class="confidence-badge confidence-high">
    🟢 HIGH CONFIDENCE
</div>
```

**Visual Effect:**
- Bright green gradient pill
- Bold black text
- Pulsing glow
- Impossible to miss

#### 🟡 MEDIUM CONFIDENCE
```css
.confidence-medium {
    background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%);
    color: #000;
    box-shadow: 0 0 12px rgba(255, 215, 0, 0.5);
    animation: pulse-badge-yellow 2s infinite;
}
```

**Visual Effect:**
- Gold-to-orange gradient
- Bold black text
- Warm glow
- Attention-getting but not overwhelming

#### ⚪ Low Confidence
```css
.confidence-low {
    background: linear-gradient(135deg, #888 0%, #666 100%);
    color: #fff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}
```

**Visual Effect:**
- Gray gradient
- White text
- No animation
- Subtle and unobtrusive

### Spacing Optimizations

**Before:**
- Card padding: `16px` all around
- Element margin: `0.5rem`
- Button padding: `8px 14px`

**After:**
- Card padding: `12px 16px` (vertical reduced)
- Element margin: `0.2rem` in cards
- Button padding: `6px 12px`
- Status text: `13px` (compact)

**Result:** ~20% more vertical space for content

### Color Palette

| Element | Color | Usage |
|---------|-------|-------|
| High Confidence Glow | `#00ff00` | Borders, glows, gradients |
| Medium Confidence Glow | `#ffd700` | Borders, glows, gradients (gold not yellow) |
| Low Confidence | `rgba(150, 150, 150, 0.6)` | Subtle gray |
| Card Border | `rgba(128, 128, 128, 0.5)` | Default 4-sided border |
| Card Shadow | `rgba(0, 0, 0, 0.2)` | Depth shadow |
| Inset Highlight | `rgba(255, 255, 255, 0.1)` | Top edge highlight |

### Animation Timings

All animations use `2s infinite` for consistent rhythm:
- `pulse-green` - Full card glow (high confidence)
- `pulse-yellow` - Full card glow (medium confidence)
- `pulse-glow-green` - Team logo glow (high confidence)
- `pulse-glow-yellow` - Team logo glow (medium confidence)
- `pulse-badge-green` - Badge glow (high confidence)
- `pulse-badge-yellow` - Badge glow (medium confidence)

### Responsive Behavior

Cards maintain their styling across all layouts:
- 2 cards per row (desktop, large displays)
- 3 cards per row (standard desktop)
- 4 cards per row (wide screens - default)

Hover effects work on desktop; static states work on mobile.

## Design Philosophy

The visual hierarchy is designed to guide the user's attention:

1. **First Glance:** High confidence cards jump out with green glow
2. **Second Glance:** Medium confidence cards show warm golden opportunity
3. **Detailed Review:** Badge, team highlighting, and metrics provide depth
4. **Low Priority:** Low confidence games fade into background

This creates an efficient betting workflow:
- Spot opportunities instantly (green)
- Evaluate decent bets (gold)
- Skip low-value games (gray)

## Accessibility Notes

- Color is reinforced with text labels (emojis + text)
- Confidence badges have high contrast (black text on bright backgrounds)
- Borders are visible regardless of theme (light/dark mode)
- Animations are subtle enough not to cause distraction
- Text remains readable at all confidence levels

## Browser Compatibility

CSS features used:
- `linear-gradient()` - All modern browsers
- `box-shadow` (multi-layer) - All modern browsers
- `border-radius` - All modern browsers
- `animation` - All modern browsers
- `transform` - All modern browsers
- `var(--custom-property)` - All modern browsers

No vendor prefixes needed for target browsers (Chrome, Firefox, Safari, Edge).
