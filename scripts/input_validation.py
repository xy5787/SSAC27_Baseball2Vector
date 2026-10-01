"""Preflight local exports without downloading, rewriting, or publishing player data."""
import hashlib
import json
import numpy as np
import pandas as pd


def validate_table(path, schema, projections=False):
    frame = pd.read_csv(path, low_memory=False)
    required = schema['projection_fields'] if projections else schema['source_fields']
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f'missing columns: {missing}')
    if projections:
        # Missing projection IDs are excluded by the existing join protocol.
        ids = frame.PlayerId.dropna().astype(str)
        if ids.duplicated().any():
            raise ValueError('duplicate projection PlayerId values')
        numeric = ['wRC+', 'WAR', 'PA']
    else:
        for column in ['Name', 'Team', 'Season']:
            if column+'.1' in frame and not frame[column].equals(frame[column+'.1']):
                raise ValueError(f'{column} and {column}.1 disagree')
        numeric = [x for x in required if x not in ['Name', 'Team']]
    for column in numeric:
        values = pd.to_numeric(frame[column], errors='raise')
        if np.isinf(values.dropna()).any():
            raise ValueError(f'{column} contains infinity')
    if not projections:
        if frame[['PlayerId','Season']].isna().any().any():
            raise ValueError('missing PlayerId or Season')
        for column in ['PlayerId', 'Season']:
            if not (frame[column] % 1 == 0).all():
                raise ValueError(f'{column} must contain integer identifiers')
        study=frame[frame.Season.between(2018,2025)]
        if set(study.Season) != set(range(2018,2026)):
            raise ValueError('batting data must cover every season 2018–2025')
        if study.duplicated(['PlayerId','Season']).any():
            raise ValueError('duplicate player-season rows; export totals across teams')
        if study.PA.isna().any() or study.loc[study.PA.gt(0), ['WAR','wRC+','Age','Name']].isna().any().any():
            raise ValueError('missing batting PA, WAR, wRC+, Age or Name')
        if study.PA.lt(0).any():
            raise ValueError('negative PA')
        if not study.PA.lt(100).any():
            raise ValueError('export appears PA-filtered; include low-PA history')
        for column in schema['rate_columns_fractional']:
            values=study[column].dropna()
            # The frozen export contains one HR/FB=2 outlier; preserve provider values.
            invalid = (values.lt(0).any() or values.quantile(.99) > 1) if column == 'HR/FB' else not values.between(0,1).all()
            if invalid:
                raise ValueError(f'{column} must use fractional rates (0.80, not 80 or 80%)')
    return dict(rows=len(frame), columns=len(frame.columns))


def inspect_inputs(root, projections=True, mode='exact'):
    if mode not in ['exact','new']:
        raise ValueError(f'Unknown input mode: {mode}')
    manifest=json.loads((root/'data/input_manifest.json').read_text())
    schema=json.loads((root/'data/input_schema.json').read_text())
    checked, errors=[], []
    for item in manifest['files']:
        projection=item['path'].startswith('ProjectionDataset/')
        if not item['required'] or (projection and not projections):
            continue
        path=root/item['path']
        if not path.is_file():
            errors.append(f"Missing {item['path']}"); continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        matches=digest==item['sha256']
        if not matches and mode=='exact':
            errors.append(f"Snapshot mismatch: {item['path']}. Use --input-mode new for a newly downloaded snapshot.")
        try:
            info=validate_table(path,schema,projection)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"Invalid {item['path']}: {exc}"); continue
        checked.append(dict(path=item['path'],sha256=digest,expected_sha256=item['sha256'],
                            matches_snapshot=matches,**info))
    if errors:
        raise SystemExit('Input preflight failed:\n'+'\n'.join(errors)+'\nSee data/DATA_RECIPE.md. No fitting started.')
    return checked
