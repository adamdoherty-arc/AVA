"""
Pattern Learner

Learns from recurring issues to:
1. Detect patterns before they become problems
2. Suggest preventive measures
3. Track issue recurrence over time
4. Generate prevention rules
5. Improve QA checks based on history
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a learned pattern."""
    id: str
    category: str
    pattern_type: str  # 'code', 'error', 'behavior'
    regex: Optional[str]
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    occurrences: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    files_affected: Set[str] = field(default_factory=set)
    prevention_rule: Optional[str] = None
    auto_generated: bool = False


@dataclass
class PatternMatch:
    """Represents a pattern match in code."""
    pattern_id: str
    file_path: str
    line_number: int
    matched_text: str
    context: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PatternLearner:
    """
    Learns patterns from QA runs and generates prevention rules.

    Tracks:
    - Recurring code issues
    - Common error patterns
    - Files with frequent problems
    - Time-based patterns (issues that appear after certain changes)
    """

    # Seed patterns to start learning from
    SEED_PATTERNS = [
        Pattern(
            id='dummy_data_random',
            category='data_quality',
            pattern_type='code',
            regex=r'random\.(uniform|randint|choice|random)\s*\(',
            description='Random data generation in production code',
            severity='high',
        ),
        Pattern(
            id='hardcoded_api_key',
            category='security',
            pattern_type='code',
            regex=r'(api_key|apikey|api_secret)\s*=\s*["\'][^"\']{20,}["\']',
            description='Hardcoded API key or secret',
            severity='critical',
        ),
        Pattern(
            id='print_statement',
            category='code_quality',
            pattern_type='code',
            regex=r'^(\s*)print\s*\(',
            description='Print statement in production code',
            severity='low',
        ),
        Pattern(
            id='todo_comment',
            category='code_quality',
            pattern_type='code',
            regex=r'#\s*(TODO|FIXME|HACK|XXX):?',
            description='TODO/FIXME comment in code',
            severity='low',
        ),
        Pattern(
            id='broad_except',
            category='error_handling',
            pattern_type='code',
            regex=r'except\s*:',
            description='Bare except clause catches all exceptions',
            severity='medium',
        ),
        Pattern(
            id='sql_injection_risk',
            category='security',
            pattern_type='code',
            regex=r'execute\s*\(\s*[f"\'].*\{.*\}',
            description='Potential SQL injection via f-string',
            severity='critical',
        ),
        Pattern(
            id='missing_await',
            category='async',
            pattern_type='code',
            regex=r'(?<!await\s)(async_\w+|fetch_\w+|get_\w+_async)\s*\(',
            description='Async function call possibly missing await',
            severity='high',
        ),
    ]

    def __init__(self, project_root: Path = None):
        """
        Initialize pattern learner.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.patterns_file = self.data_dir / "learned_patterns.json"
        self.matches_file = self.data_dir / "pattern_matches.jsonl"
        self.stats_file = self.data_dir / "pattern_stats.json"

        self.patterns: Dict[str, Pattern] = {}
        self.recent_matches: List[PatternMatch] = []

        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from file or initialize with seeds."""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    for p_data in data.get('patterns', []):
                        pattern = Pattern(
                            id=p_data['id'],
                            category=p_data['category'],
                            pattern_type=p_data['pattern_type'],
                            regex=p_data.get('regex'),
                            description=p_data['description'],
                            severity=p_data['severity'],
                            occurrences=p_data.get('occurrences', 0),
                            first_seen=datetime.fromisoformat(p_data['first_seen']) if p_data.get('first_seen') else None,
                            last_seen=datetime.fromisoformat(p_data['last_seen']) if p_data.get('last_seen') else None,
                            files_affected=set(p_data.get('files_affected', [])),
                            prevention_rule=p_data.get('prevention_rule'),
                            auto_generated=p_data.get('auto_generated', False),
                        )
                        self.patterns[pattern.id] = pattern
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")

        # Add seed patterns if not present
        for seed in self.SEED_PATTERNS:
            if seed.id not in self.patterns:
                self.patterns[seed.id] = seed

    def _save_patterns(self) -> None:
        """Save patterns to file."""
        try:
            data = {
                'patterns': [
                    {
                        'id': p.id,
                        'category': p.category,
                        'pattern_type': p.pattern_type,
                        'regex': p.regex,
                        'description': p.description,
                        'severity': p.severity,
                        'occurrences': p.occurrences,
                        'first_seen': p.first_seen.isoformat() if p.first_seen else None,
                        'last_seen': p.last_seen.isoformat() if p.last_seen else None,
                        'files_affected': list(p.files_affected),
                        'prevention_rule': p.prevention_rule,
                        'auto_generated': p.auto_generated,
                    }
                    for p in self.patterns.values()
                ],
                'updated_at': datetime.utcnow().isoformat(),
            }
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")

    def scan_codebase(self) -> List[PatternMatch]:
        """
        Scan the entire codebase for pattern matches.

        Returns:
            List of all pattern matches found.
        """
        all_matches = []

        for py_file in self.project_root.rglob('*.py'):
            if self._should_skip(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(py_file.relative_to(self.project_root))

                file_matches = self._scan_file(content, relative_path)
                all_matches.extend(file_matches)

            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")

        # Update pattern statistics
        self._update_statistics(all_matches)

        # Log matches
        self._log_matches(all_matches)

        # Learn new patterns from matches
        self._learn_from_matches(all_matches)

        # Save updated patterns
        self._save_patterns()

        self.recent_matches = all_matches
        return all_matches

    def _scan_file(self, content: str, file_path: str) -> List[PatternMatch]:
        """Scan a single file for pattern matches."""
        matches = []
        lines = content.split('\n')

        for pattern in self.patterns.values():
            if not pattern.regex:
                continue

            try:
                regex_matches = list(re.finditer(pattern.regex, content, re.MULTILINE))

                for match in regex_matches:
                    line_num = content[:match.start()].count('\n') + 1

                    # Get context (surrounding lines)
                    start_line = max(0, line_num - 2)
                    end_line = min(len(lines), line_num + 2)
                    context = '\n'.join(lines[start_line:end_line])

                    matches.append(PatternMatch(
                        pattern_id=pattern.id,
                        file_path=file_path,
                        line_number=line_num,
                        matched_text=match.group()[:100],
                        context=context[:200],
                    ))

            except re.error as e:
                logger.warning(f"Invalid regex for pattern {pattern.id}: {e}")

        return matches

    def _update_statistics(self, matches: List[PatternMatch]):
        """Update pattern statistics based on matches."""
        now = datetime.utcnow()

        # Group matches by pattern
        by_pattern = defaultdict(list)
        for match in matches:
            by_pattern[match.pattern_id].append(match)

        # Update each pattern
        for pattern_id, pattern_matches in by_pattern.items():
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.occurrences += len(pattern_matches)
                pattern.last_seen = now

                if not pattern.first_seen:
                    pattern.first_seen = now

                for match in pattern_matches:
                    pattern.files_affected.add(match.file_path)

    def _log_matches(self, matches: List[PatternMatch]):
        """Log matches to JSONL file."""
        try:
            with open(self.matches_file, 'a', encoding='utf-8') as f:
                for match in matches:
                    entry = {
                        'timestamp': match.timestamp.isoformat(),
                        'pattern_id': match.pattern_id,
                        'file_path': match.file_path,
                        'line_number': match.line_number,
                        'matched_text': match.matched_text,
                    }
                    f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log matches: {e}")

    def _learn_from_matches(self, matches: List[PatternMatch]):
        """
        Learn new patterns from recurring matches.

        Analyzes:
        - Files with multiple issues
        - Correlated patterns
        - New patterns not in seed list
        """
        # Find files with multiple different issues
        files_with_issues = defaultdict(set)
        for match in matches:
            files_with_issues[match.file_path].add(match.pattern_id)

        # Identify "problem files" (files with 3+ different issues)
        problem_files = {
            f: patterns for f, patterns in files_with_issues.items()
            if len(patterns) >= 3
        }

        if problem_files:
            logger.info(f"Found {len(problem_files)} files with multiple issues")

            # Create a meta-pattern for problem files
            self._create_problem_file_pattern(problem_files)

        # Analyze common code structures that lead to issues
        self._analyze_code_structures(matches)

    def _create_problem_file_pattern(self, problem_files: Dict[str, Set[str]]):
        """Create a pattern for files that frequently have issues."""
        pattern_id = 'problem_file_cluster'

        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = Pattern(
                id=pattern_id,
                category='meta',
                pattern_type='behavior',
                regex=None,  # Not regex-based
                description='File frequently has multiple issues',
                severity='medium',
                auto_generated=True,
                prevention_rule='Consider refactoring files that frequently trigger multiple issues',
            )

        # Update files affected
        self.patterns[pattern_id].files_affected.update(problem_files.keys())
        self.patterns[pattern_id].occurrences = len(problem_files)
        self.patterns[pattern_id].last_seen = datetime.utcnow()

    def _analyze_code_structures(self, matches: List[PatternMatch]):
        """Analyze code structures that commonly lead to issues."""
        # Group matches by context similarity
        context_patterns = defaultdict(list)

        for match in matches:
            # Extract structural elements from context
            structure = self._extract_structure(match.context)
            if structure:
                context_patterns[structure].append(match)

        # Find recurring structural patterns
        for structure, struct_matches in context_patterns.items():
            if len(struct_matches) >= 3:
                # Create a new learned pattern
                self._create_structural_pattern(structure, struct_matches)

    def _extract_structure(self, context: str) -> Optional[str]:
        """Extract structural pattern from context."""
        # Normalize the context
        # Remove variable names, strings, numbers
        normalized = re.sub(r'\b[a-z_][a-z0-9_]*\b', 'VAR', context)
        normalized = re.sub(r'"[^"]*"', 'STR', normalized)
        normalized = re.sub(r"'[^']*'", 'STR', normalized)
        normalized = re.sub(r'\b\d+\b', 'NUM', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        if len(normalized) < 20:
            return None

        return normalized[:100]

    def _create_structural_pattern(self, structure: str, matches: List[PatternMatch]):
        """Create a pattern from structural analysis."""
        # Generate a unique ID
        pattern_id = f"learned_{hash(structure) % 10000:04d}"

        if pattern_id not in self.patterns:
            # Try to create a regex from the structure
            regex = self._structure_to_regex(structure)

            self.patterns[pattern_id] = Pattern(
                id=pattern_id,
                category='learned',
                pattern_type='code',
                regex=regex,
                description=f'Learned pattern from {len(matches)} occurrences',
                severity='low',
                auto_generated=True,
                first_seen=datetime.utcnow(),
            )

        # Update statistics
        self.patterns[pattern_id].occurrences += len(matches)
        self.patterns[pattern_id].last_seen = datetime.utcnow()
        for match in matches:
            self.patterns[pattern_id].files_affected.add(match.file_path)

    def _structure_to_regex(self, structure: str) -> Optional[str]:
        """Convert structural pattern to regex."""
        try:
            # Escape special regex characters
            escaped = re.escape(structure)

            # Replace placeholders with patterns
            regex = escaped.replace('VAR', r'\w+')
            regex = regex.replace('STR', r'["\'][^"\']*["\']')
            regex = regex.replace('NUM', r'\d+')

            # Validate the regex
            re.compile(regex)
            return regex

        except re.error:
            return None

    def generate_prevention_rules(self) -> List[Dict[str, Any]]:
        """Generate prevention rules based on learned patterns."""
        rules = []

        for pattern in self.patterns.values():
            if pattern.occurrences >= 5 and pattern.severity in ('high', 'critical'):
                rule = {
                    'pattern_id': pattern.id,
                    'description': pattern.description,
                    'severity': pattern.severity,
                    'occurrences': pattern.occurrences,
                    'files_affected': len(pattern.files_affected),
                    'prevention_rule': pattern.prevention_rule or self._generate_rule(pattern),
                    'auto_generated': pattern.auto_generated,
                }
                rules.append(rule)

        return sorted(rules, key=lambda r: r['occurrences'], reverse=True)

    def _generate_rule(self, pattern: Pattern) -> str:
        """Generate a prevention rule for a pattern."""
        rules_map = {
            'dummy_data_random': 'Use data from actual API sources instead of random generation',
            'hardcoded_api_key': 'Store secrets in environment variables or secret manager',
            'print_statement': 'Use logging module instead of print statements',
            'todo_comment': 'Create tickets for TODOs and resolve before merging',
            'broad_except': 'Catch specific exceptions and handle appropriately',
            'sql_injection_risk': 'Use parameterized queries instead of f-strings',
            'missing_await': 'Ensure all async functions are awaited',
        }

        return rules_map.get(pattern.id, f'Review and fix {pattern.category} issues')

    def get_hot_spots(self) -> List[Dict[str, Any]]:
        """Get files that are frequent sources of issues."""
        file_issues = defaultdict(lambda: {'count': 0, 'patterns': set(), 'severity_score': 0})

        severity_weights = {'low': 1, 'medium': 2, 'high': 5, 'critical': 10}

        for pattern in self.patterns.values():
            weight = severity_weights.get(pattern.severity, 1)
            for file_path in pattern.files_affected:
                file_issues[file_path]['count'] += pattern.occurrences
                file_issues[file_path]['patterns'].add(pattern.id)
                file_issues[file_path]['severity_score'] += pattern.occurrences * weight

        # Sort by severity score
        hot_spots = [
            {
                'file': file_path,
                'issue_count': data['count'],
                'pattern_count': len(data['patterns']),
                'severity_score': data['severity_score'],
                'patterns': list(data['patterns']),
            }
            for file_path, data in file_issues.items()
        ]

        return sorted(hot_spots, key=lambda x: x['severity_score'], reverse=True)[:20]

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall pattern statistics."""
        total_patterns = len(self.patterns)
        total_occurrences = sum(p.occurrences for p in self.patterns.values())
        auto_generated = sum(1 for p in self.patterns.values() if p.auto_generated)

        by_severity = defaultdict(int)
        by_category = defaultdict(int)

        for pattern in self.patterns.values():
            by_severity[pattern.severity] += pattern.occurrences
            by_category[pattern.category] += pattern.occurrences

        return {
            'total_patterns': total_patterns,
            'total_occurrences': total_occurrences,
            'auto_generated_patterns': auto_generated,
            'by_severity': dict(by_severity),
            'by_category': dict(by_category),
            'hot_spots': self.get_hot_spots()[:5],
        }

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_parts = [
            'venv', '.venv', 'node_modules', '__pycache__',
            'tests', 'test_', '.git', 'dist', 'build',
        ]
        path_str = str(file_path)
        return any(p in path_str for p in skip_parts)
