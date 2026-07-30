# FusionRS correction and takedown policy

Requests concerning rights, privacy, corrupted records, duplicate leakage, or
unsafe content must identify the affected `sample_id` or upstream record.

Verified issues are handled by:

1. marking the record with a tombstone in the next manifest version;
2. removing any hosted derivative when applicable;
3. recording a non-sensitive reason and effective date in the changelog;
4. issuing a major split version when component membership or held-out
   integrity changes; and
5. marking benchmark results based on the superseded split as stale.

The final public repository must replace this paragraph with a maintainer
address and response-time target before release.

