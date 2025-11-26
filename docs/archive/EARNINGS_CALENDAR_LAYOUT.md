# Earnings Calendar - Page Layout & Design

## Visual Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         📅 EARNINGS CALENDAR                                    │
│                Track upcoming earnings, historical performance, and expected    │
│                                     moves                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌─────────────────────────────────────────────────────────────┐
│              │  │                                                             │
│  FILTERS     │  │                   📊 ANALYTICS                             │
│              │  │                                                             │
│ Date Range   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────┐
│ ◉ This Week  │  │  │ Total   │  │ Pending │  │ Beat    │  │  Avg    │  │Beat│
│ ○ Next Week  │  │  │ Events  │  │         │  │  Rate   │  │Surprise │  │/   │
│ ○ This Month │  │  │   42    │  │   15    │  │  72.5%  │  │  +3.2%  │  │Miss│
│ ○ Next Month │  │  │         │  │         │  │         │  │         │  │    │
│ ○ Custom     │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └────┘
│              │  │                                                             │
│ Time of Day  │  └─────────────────────────────────────────────────────────────┘
│ ▼ All        │
│              │  ┌─────────────────────────────────────────────────────────────┐
│ Sector       │  │                                                             │
│ ▼ All Sectors│  │            📅 Calendar View  |  📋 List View  |  📈 Analysis│
│              │  │                                                             │
│              │  │  ┌─ CALENDAR VIEW ──────────────────────────────────────┐  │
│ ─────────    │  │  │                October 2025                          │  │
│              │  │  │                                                       │  │
│ DATA SYNC    │  │  │  Mon    Tue    Wed    Thu    Fri    Sat    Sun      │  │
│              │  │  │  ────────────────────────────────────────────────────│  │
│ Last Sync:   │  │  │         1      2      3      4      5      6         │  │
│ Oct 28, 2025 │  │  │                                                       │  │
│ 10:30 AM     │  │  │   7      8      9     10     11     12     13        │  │
│              │  │  │         AAPL                MSFT                      │  │
│ ┌──────────┐ │  │  │         AMC                 AMC                       │  │
│ │ 🔄 Sync  │ │  │  │                                                       │  │
│ │Robinhood │ │  │  │  14     15     16     17     18     19     20        │  │
│ └──────────┘ │  │  │  GOOGL  AMZN   NVDA                                  │  │
│              │  │  │  AMC    AMC    AMC                                    │  │
│              │  │  │  TSLA                                                 │  │
│              │  │  │  AMC                                                  │  │
│              │  │  │                                                       │  │
│              │  │  │  21     22     23     24     25     26     27        │  │
│              │  │  │         AMD    NFLX   INTC                            │  │
│              │  │  │         AMC    AMC    AMC                             │  │
│              │  │  │                                                       │  │
│              │  │  │  28     29     30     31                              │  │
│              │  │  │                                                       │  │
│              │  │  └───────────────────────────────────────────────────────┘  │
│              │  │                                                             │
│              │  │  ┌─ LIST VIEW ──────────────────────────────────────────┐  │
│              │  │  │                                                       │  │
│              │  │  │ Symbol  Company      Date    Time  EPS   EPS   Sur%  │  │
│              │  │  │                                     Est   Act         │  │
│              │  │  │ ─────────────────────────────────────────────────────│  │
│              │  │  │ AAPL    Apple Inc    10/08  AMC   2.10  2.15  +2.4% │  │ ← Green
│              │  │  │ MSFT    Microsoft    10/11  AMC   2.65  2.55  -3.8% │  │ ← Red
│              │  │  │ GOOGL   Alphabet     10/14  AMC   1.45  ---   Pend  │  │ ← Gray
│              │  │  │ AMZN    Amazon       10/15  AMC   1.20  ---   Pend  │  │
│              │  │  │ NVDA    NVIDIA       10/16  AMC   5.50  ---   Pend  │  │
│              │  │  │ TSLA    Tesla        10/17  AMC   0.85  ---   Pend  │  │
│              │  │  │                                                       │  │
│              │  │  │                    ┌───────────────┐                 │  │
│              │  │  │                    │ 📥 Export CSV │                 │  │
│              │  │  │                    └───────────────┘                 │  │
│              │  │  └───────────────────────────────────────────────────────┘  │
│              │  │                                                             │
│              │  │  ┌─ HISTORICAL ANALYSIS ─────────────────────────────────┐  │
│              │  │  │                                                       │  │
│              │  │  │  Select Symbol:  [AAPL    ▼]                         │  │
│              │  │  │                                                       │  │
│              │  │  │     AAPL - Historical Earnings Performance            │  │
│              │  │  │  │                                                    │  │
│              │  │  │ $│     ┌─┐                                            │  │
│              │  │  │ 2│     │█│    ┌─┐                                     │  │
│              │  │  │ .│  ┌─┐│█│ ┌─┐│█│                                     │  │
│              │  │  │ 5│  │█││█│ │█││█│                                     │  │
│              │  │  │ 0│  │█││█│ │█││█│  ▬ Actual EPS                       │  │
│              │  │  │  │  └─┘└─┘ └─┘└─┘  ● Estimated EPS                   │  │
│              │  │  │  └──────────────────────────────────>                │  │
│              │  │  │     Q1   Q2   Q3   Q4   Q1   Q2                      │  │
│              │  │  │    2024 2024 2024 2024 2025 2025                     │  │
│              │  │  │                                                       │  │
│              │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │  │
│              │  │  │  │ Beat    │  │  Avg    │  │  Avg    │              │  │
│              │  │  │  │  Rate   │  │Surprise │  │  Price  │              │  │
│              │  │  │  │  8/10   │  │  +4.2%  │  │  Move   │              │  │
│              │  │  │  │  80%    │  │         │  │  +5.3%  │              │  │
│              │  │  │  └─────────┘  └─────────┘  └─────────┘              │  │
│              │  │  │                                                       │  │
│              │  │  └───────────────────────────────────────────────────────┘  │
│              │  │                                                             │
│              │  └─────────────────────────────────────────────────────────────┘
│              │
└──────────────┘
```

## Color Scheme

### Status Colors
- **Green (`#10b981`)**: Beat estimates - positive performance
- **Red (`#ef4444`)**: Missed estimates - negative performance
- **Gray (`#6b7280`)**: Pending - no results yet
- **Yellow (`#fbbf24`)**: Inline - met estimates exactly

### Time Badges
- **Blue (`#3b82f6`)**: BMO (Before Market Open)
- **Purple (`#8b5cf6`)**: AMC (After Market Close)

### Metric Cards
- **Gradient Purple/Blue**: Analytics cards at top
- **White Background**: Main content areas
- **Light Gray (`#f3f4f6`)**: Calendar day boxes

## Component Breakdown

### 1. Header Section
```
┌───────────────────────────────────────┐
│  📅 EARNINGS CALENDAR                 │
│  Subtitle text explaining the page    │
└───────────────────────────────────────┘
```
- Large title with icon
- Descriptive subtitle
- Clean white background

### 2. Analytics Cards (Responsive Grid)
```
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Total │ │Pend- │ │Beat  │ │Avg   │ │Beat/ │
│Events│ │ing   │ │Rate  │ │Surp. │ │Miss  │
│  42  │ │  15  │ │72.5% │ │+3.2% │ │27/10 │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘
```
- 5 columns on desktop, stack on mobile
- Large numbers with labels
- Delta indicators where relevant

### 3. Sidebar Filters
```
┌─────────────────┐
│ Date Range      │
│ ◉ This Week     │
│ ○ Next Week     │
│ ○ This Month    │
│ ○ Next Month    │
│ ○ Custom        │
├─────────────────┤
│ Time of Day     │
│ ▼ All           │
├─────────────────┤
│ Sector          │
│ ▼ All Sectors   │
├─────────────────┤
│ [🔄 Sync Data]  │
└─────────────────┘
```
- Collapsible sections
- Radio buttons for exclusive choices
- Dropdowns for multiple options
- Action button at bottom

### 4. Tab Navigation
```
┌─────────────┬─────────────┬─────────────┐
│📅 Calendar  │📋 List View │📈 Historical│
├─────────────┴─────────────┴─────────────┤
│                                          │
│         Active tab content here          │
│                                          │
└──────────────────────────────────────────┘
```
- Three main views
- Icons + labels
- Active tab highlighted

### 5. Calendar Grid
```
  Mon    Tue    Wed    Thu    Fri    Sat    Sun
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  1   │  2   │  3   │  4   │  5   │  6   │  7   │
│      │ AAPL │      │      │ MSFT │      │      │
│      │ AMC  │      │      │ AMC  │      │      │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│  8   │  9   │  10  │  11  │  12  │  13  │  14  │
│GOOGL │      │ AMZN │ NVDA │      │      │      │
│AMC   │      │ AMC  │ AMC  │      │      │      │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```
- 7 columns (days of week)
- Variable rows (weeks in month)
- Day number at top
- Events listed below
- Max 3 events visible, "+X more" for overflow

### 6. List/Table View
```
┌─────────────────────────────────────────────────────┐
│ ▲ Symbol │ Company    │ Date  │ Time │ EPS Est │ ...│
├──────────┼────────────┼───────┼──────┼─────────┼────┤
│   AAPL   │ Apple Inc  │ 10/08 │ AMC  │  2.10   │ ...│ ← Green
│   MSFT   │ Microsoft  │ 10/11 │ AMC  │  2.65   │ ...│ ← Red
│   GOOGL  │ Alphabet   │ 10/14 │ AMC  │  1.45   │ ...│ ← Gray
└──────────┴────────────┴───────┴──────┴─────────┴────┘
```
- Sortable columns (click header)
- Row highlighting by status
- Hover effects
- Responsive width
- Fixed header on scroll

### 7. Historical Chart
```
    EPS ($)
    3.0 ┤     ┌─┐
    2.5 ┤  ┌─┐│█│
    2.0 ┤  │█││█│
    1.5 ┤  │█││█│
    1.0 ┤  │█││█│
    0.5 ┤  └─┘└─┘
        └─────────────>
         Q1  Q2  Q3  Q4
```
- Bar chart for actual EPS
- Line overlay for estimates
- X-axis: quarters
- Y-axis: dollar amount
- Hover tooltips with details

### 8. Sync Progress
```
┌─────────────────────────────┐
│  Syncing earnings data...   │
│  ████████████░░░░░░ 60%     │
│  NVDA... (30/50)            │
└─────────────────────────────┘
```
- Progress bar
- Current symbol
- Count (current/total)
- Success/error message when done

## Responsive Breakpoints

### Desktop (>1200px)
- Sidebar: 250px fixed width
- Main content: Remaining width
- Analytics: 5 columns
- Calendar: Full month grid
- Table: All columns visible

### Tablet (768px - 1200px)
- Sidebar: Collapsible drawer
- Main content: Full width
- Analytics: 3 columns, wrap to 2 rows
- Calendar: Full month, smaller text
- Table: Horizontal scroll

### Mobile (<768px)
- Sidebar: Hamburger menu
- Main content: Full width
- Analytics: 2 columns, stack
- Calendar: Week view instead of month
- Table: Card view (stacked rows)

## Interactive Elements

### Hover Effects
- **Calendar Events**: Lighten background, show tooltip
- **Table Rows**: Highlight row, change cursor
- **Buttons**: Darken background, scale slightly
- **Cards**: Subtle shadow increase

### Click Actions
- **Calendar Event**: Open detail modal/drawer
- **Table Row**: Expand inline details or navigate
- **Symbol**: Jump to historical analysis
- **Export**: Download CSV immediately
- **Sync**: Start sync, show progress

### Tooltips
- **Metrics**: Explain calculation
- **Status Badges**: Show beat/miss amount
- **Expected Move**: Show IV and calculation
- **Symbols**: Company name if abbreviated

## Data Loading States

### Initial Load
```
┌─────────────────────┐
│    Loading...       │
│     ⏳              │
└─────────────────────┘
```

### Empty State
```
┌─────────────────────────────┐
│  No earnings events found   │
│                             │
│  Try:                       │
│  • Adjusting filters        │
│  • Syncing from Robinhood   │
│  • Adding date range        │
└─────────────────────────────┘
```

### Error State
```
┌─────────────────────────────┐
│  ⚠️ Error loading data      │
│                             │
│  Database connection failed │
│  [Retry]                    │
└─────────────────────────────┘
```

## Accessibility

- **ARIA Labels**: All interactive elements
- **Keyboard Navigation**: Tab through filters, table
- **Screen Reader**: Descriptive text for status
- **Color Blind Safe**: Not relying only on color
  - Beat: Green + "↑" symbol
  - Miss: Red + "↓" symbol
  - Pending: Gray + "•" symbol
- **High Contrast Mode**: Support system preference

## Performance Considerations

### Optimization
- **Pagination**: Table shows 50 rows, load more on demand
- **Virtual Scrolling**: For large datasets
- **Lazy Loading**: Charts load when tab is visible
- **Caching**: Cache API responses for 5 minutes
- **Debouncing**: Filter changes debounced 300ms

### Loading Strategy
1. Load analytics first (fast)
2. Load upcoming events (medium)
3. Load historical data (slow, on-demand)
4. Render progressively

## Future Enhancements

### Phase 2
- [ ] **Earnings Alerts**: Set reminders for specific stocks
- [ ] **Pre/Post Analysis**: Automated before/after comparison
- [ ] **Sector Heatmap**: Visual sector performance
- [ ] **Options Flow**: Track unusual options activity

### Phase 3
- [ ] **AI Predictions**: ML model for earnings surprises
- [ ] **Transcript Analysis**: Parse earnings call text
- [ ] **Social Sentiment**: Track Twitter/Reddit mentions
- [ ] **Analyst Changes**: Rating upgrades/downgrades

### Phase 4
- [ ] **Live Updates**: WebSocket for real-time data
- [ ] **Mobile App**: Native iOS/Android
- [ ] **Notifications**: Push alerts for earnings
- [ ] **Portfolio Integration**: Show only my holdings

This layout provides a comprehensive, professional earnings calendar with all the features needed for serious options trading and wheel strategy execution.
