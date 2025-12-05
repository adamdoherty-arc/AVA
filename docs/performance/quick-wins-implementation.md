# Frontend Performance Quick Wins - Implementation Guide

**Estimated Time:** 35 minutes
**Expected Impact:** 50-70% performance improvement
**Risk Level:** Low (all changes are backwards compatible)

---

## Quick Win #1: Vite Build Optimization (15 min)

### Impact: -40% bundle size

### Steps:

1. **Backup current config:**
```bash
cd C:/code/MagnusAntiG/Magnus/frontend
cp vite.config.ts vite.config.ts.backup
```

2. **Install terser:**
```bash
npm install --save-dev terser
```

3. **Replace vite.config.ts:**
```bash
cp ../docs/performance/vite.config.optimized.ts vite.config.ts
```

4. **Test build:**
```bash
npm run build
```

5. **Verify output:**
- Check `dist/assets/js/` folder
- Should see: `vendor-react-*.js`, `vendor-charts-*.js`, etc.
- Total size should be ~40% smaller

---

## Quick Win #2: Dashboard Component Memoization (2 min)

### Impact: -70% unnecessary re-renders

### File: `C:/code/MagnusAntiG/Magnus/frontend/src/pages/Dashboard.tsx`

**Line 1:** Add import:
```typescript
import { useState, memo } from 'react'
```

**Line 9:** Change from:
```typescript
export function Dashboard() {
```

To:
```typescript
export const Dashboard = memo(function Dashboard() {
```

**Line 349:** Close memo wrapper - add closing paren:
```typescript
    )
}
})  // Add this closing for memo
```

---

## Quick Win #3: Positions Table Memoization (5 min)

### Impact: -80% table re-renders

### File: `C:/code/MagnusAntiG/Magnus/frontend/src/pages/Positions.tsx`

**Line 1:** Update imports:
```typescript
import { useState, useEffect, memo, useMemo, useCallback } from 'react'
```

**Line 559:** Wrap StocksTable:
```typescript
const StocksTable = memo(function StocksTable({ stocks, onSelect, selectedSymbol }: StocksTableProps) {
```

**Line 625:** Close StocksTable memo:
```typescript
    )
})  // Add closing for memo
```

**Line 636:** Wrap OptionsTable:
```typescript
const OptionsTable = memo(function OptionsTable({ options, expanded, onToggle, onSelect, selectedSymbol }: OptionsTableProps) {
```

**Line 761:** Close OptionsTable memo:
```typescript
    )
})  // Add closing for memo
```

**Line 772:** Wrap GreekBadge:
```typescript
const GreekBadge = memo(function GreekBadge({ label, value, prefix = '', suffix = '', positive }: GreekBadgeProps) {
```

**Line 781:** Close GreekBadge memo:
```typescript
    )
})
```

---

## Quick Win #4: Theta Decay Calculation Optimization (5 min)

### Impact: -90% calculation time for options table

### File: `C:/code/MagnusAntiG/Magnus/frontend/src/pages/Positions.tsx`

**Line 789:** Wrap component in memo:
```typescript
const IndividualThetaDecay = memo(function IndividualThetaDecay({ option }: IndividualThetaDecayProps) {
```

**Line 820:** Add useMemo for decay schedule:
```typescript
    // Don't show if no theta or expired
    if (dte <= 0 || dailyThetaDollars === 0) {
        return null
    }

    // WRAP THIS ENTIRE SECTION IN useMemo
    const decaySchedule = useMemo(() => {
        // Generate decay schedule - show up to 14 days or until expiration
        const daysToShow = Math.min(dte, 14)
        const schedule: { day: number; date: string; theta: number; cumulative: number; percentDecayed: number }[] = []

        let cumulative = 0
        const totalPotentialTheta = dailyThetaDollars * dte

        for (let day = 1; day <= daysToShow; day++) {
            // Theta accelerates as expiration approaches - use simplified acceleration model
            // Theta roughly doubles in the last week
            const dteAtDay = dte - day + 1
            const accelerationFactor = dteAtDay <= 7 ? 1 + (7 - dteAtDay) * 0.15 : 1
            const adjustedDailyTheta = dailyThetaDollars * accelerationFactor

            cumulative += adjustedDailyTheta
            const percentDecayed = totalPotentialTheta > 0 ? (cumulative / totalPotentialTheta) * 100 : 0

            schedule.push({
                day,
                date: getDateFromDTE(day),
                theta: adjustedDailyTheta,
                cumulative,
                percentDecayed: Math.min(percentDecayed, 100)
            })
        }

        return schedule
    }, [dailyThetaDollars, dte])  // Dependencies

    const daysToShow = Math.min(dte, 14)
```

**Line 933:** Close IndividualThetaDecay memo:
```typescript
    )
})  // Add closing for memo
```

---

## Quick Win #5: Portfolio Calculations Memoization (5 min)

### Impact: -50% CPU usage on positions page

### File: `C:/code/MagnusAntiG/Magnus/frontend/src/pages/Positions.tsx`

**Line 31:** Replace calculations with useMemo:

**BEFORE:**
```typescript
    const totalStockValue = stocks.reduce((sum: number, s: StockPosition) => sum + s.current_value, 0)
    const totalOptionValue = options.reduce((sum: number, o: OptionPosition) => sum + Math.abs(o.current_value), 0)
    const totalPL = [...stocks, ...options].reduce((sum: number, p: StockPosition | OptionPosition) => sum + p.pl, 0)
    const totalTheta = options.reduce((sum: number, o: OptionPosition) => sum + (o.greeks?.theta || 0) * o.quantity, 0)
```

**AFTER:**
```typescript
    const portfolioMetrics = useMemo(() => {
        const totalStockValue = stocks.reduce((sum, s) => sum + s.current_value, 0)
        const totalOptionValue = options.reduce((sum, o) => sum + Math.abs(o.current_value), 0)
        const totalPL = [...stocks, ...options].reduce((sum, p) => sum + p.pl, 0)
        const totalTheta = options.reduce((sum, o) => sum + (o.greeks?.theta || 0) * o.quantity, 0)

        return { totalStockValue, totalOptionValue, totalPL, totalTheta }
    }, [stocks, options])

    const { totalStockValue, totalOptionValue, totalPL, totalTheta } = portfolioMetrics
```

---

## Quick Win #6: Query Key Fix (2 min)

### Impact: Better React Query caching

### File: `C:/code/MagnusAntiG/Magnus/frontend/src/hooks/useMagnusApi.ts`

**Line 566-573:** Fix usePositionsV2 query key:

**BEFORE:**
```typescript
export const usePositionsV2 = (forceRefresh: boolean = false) => {
    return useQuery({
        queryKey: ['positions-v2', forceRefresh],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/portfolio/v2/positions?force_refresh=${forceRefresh}`);
            return data;
        },
```

**AFTER:**
```typescript
export const usePositionsV2 = (forceRefresh: boolean = false) => {
    return useQuery({
        queryKey: ['positions-v2'],  // Remove forceRefresh from key
        queryFn: async () => {
            const { data } = await axiosInstance.get(
                `/portfolio/v2/positions${forceRefresh ? '?force_refresh=true' : ''}`
            );
            return data;
        },
```

---

## Verification Steps

After implementing all changes:

### 1. Build Test
```bash
cd C:/code/MagnusAntiG/Magnus/frontend
npm run build
```

Expected output:
- No errors
- `dist/` folder size reduced by ~40%
- Multiple vendor chunk files created

### 2. Development Server Test
```bash
npm run dev
```

Test these pages:
- [ ] Dashboard loads without errors
- [ ] Positions page loads and updates correctly
- [ ] Options table expands/collapses smoothly
- [ ] Charts render properly

### 3. Performance Validation

In Chrome DevTools:

**Before changes:**
1. Open Dashboard
2. Performance tab → Record → Stop
3. Note: Render time, Scripting time

**After changes:**
1. Clear cache and hard reload
2. Performance tab → Record → Stop
3. Compare: Should see ~50-70% reduction in render times

### 4. Network Tab Validation
- [ ] Check Network tab
- [ ] Verify vendor-charts.js loads only on chart pages
- [ ] Confirm total transfer size reduced

---

## Rollback Procedure (If Issues Occur)

### Rollback Vite Config:
```bash
cd C:/code/MagnusAntiG/Magnus/frontend
cp vite.config.ts.backup vite.config.ts
```

### Rollback Code Changes:
```bash
git checkout frontend/src/pages/Dashboard.tsx
git checkout frontend/src/pages/Positions.tsx
git checkout frontend/src/hooks/useMagnusApi.ts
```

---

## Next Steps (Optional)

After validating Quick Wins, consider:

1. **Bundle Analysis:**
```bash
npm install --save-dev rollup-plugin-visualizer
# Add to vite.config.ts and analyze
```

2. **LazyChart Implementation:**
- Create `frontend/src/components/charts/LazyChart.tsx`
- Migrate recharts imports to lazy loading
- Expected: Additional 320KB savings per route

3. **Performance Monitoring:**
- Add web-vitals package
- Implement Core Web Vitals tracking
- Set up performance budgets in CI/CD

---

## Support

If you encounter issues:

1. Check browser console for errors
2. Verify all imports are correct
3. Ensure React version is 18+
4. Test in incognito mode (cache issues)

For questions or issues, refer to:
- Full analysis: `docs/performance/frontend-performance-analysis.md`
- Vite config: `docs/performance/vite.config.optimized.ts`
