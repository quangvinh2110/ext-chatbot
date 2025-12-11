import os
import random
import collections
from typing import Optional, Dict, Any, List, Literal, Set, Tuple
from openai import OpenAI

from src.prompts import GROUP_COLUMNS_TEMPLATE, MERGE_COLUMN_GROUPS_TEMPLATE
from src.processor.table.utils import (
    format_table_data_snippet_with_header,
    extract_json,
    compute_lcs_length,
)


class ColumnGrouper:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY")
        )
        if not (model or os.getenv("LLM_MODEL")):
            raise ValueError("LLM_MODEL not set. Please provide a model name or set LLM_MODEL environment variable.")
        self.model: str = str(model) if model else str(os.getenv("LLM_MODEL"))


    def __call__(
        self,
        data_rows: List[List[Any]],
        formatted_header: List[str],
        sheet_name: Optional[str] = None,
        method: Literal["rule", "semantic", "hybrid"] = "rule",
        **kwargs,
    ) -> List[List[str]]:
        if method == "rule":
            return self.group_columns_by_rule(
                data_rows=data_rows,
                formatted_header=formatted_header,
                sample_size=kwargs.get("sample_size", 20),
                lcs_threshold=kwargs.get("lcs_threshold", 5),
                min_group_size=kwargs.get("min_group_size", 3),
                max_group_size=kwargs.get("max_group_size", 8),
            )
        elif method == "semantic":
            return self.group_columns_by_semantic(
                data_rows=data_rows, 
                formatted_header=formatted_header, 
                sheet_name=sheet_name,
                max_retries=kwargs.get("max_retries", 3),
                sample_size=kwargs.get("sample_size", 5),
            )
        elif method == "hybrid":
            return self.group_columns_by_hybrid(
                data_rows=data_rows,
                formatted_header=formatted_header,
                sheet_name=sheet_name,
                sample_size=kwargs.get("sample_size", 20),
                lcs_threshold=kwargs.get("lcs_threshold", 5),
                min_group_size=kwargs.get("min_group_size", 3),
                max_group_size=kwargs.get("max_group_size", 10),
                max_retries=kwargs.get("max_retries", 3),
            )
        else:
            raise ValueError(f"Invalid method: {method}")



    def _are_columns_related(
        self,
        col1_data: List[Any], 
        col2_data: List[Any],
        lcs_threshold: int = 5,
    ) -> bool:
        """
        Checks if two columns are related based on the LCS criteria.
        """
        for val1, val2 in zip(col1_data, col2_data):
            str_val1 = str(val1).lower()
            str_val2 = str(val2).lower()
            
            if not str_val1 or not str_val2:
                continue
                
            if compute_lcs_length(str_val1, str_val2) >= lcs_threshold:
                continue
            elif str_val1 in str_val2 or str_val2 in str_val1:
                continue
            else:
                return False
            
        return True

    
    def _merge_groups_by_rule(
        self, 
        groups: List[List[str]], 
        min_size: int = 3, 
        max_size: int = 8,
    ) -> List[List[str]]:
        """
        Merges small adjacent groups into larger groups.
        """
        if not groups: 
            return []
        merged_groups = []
        i, n = 0, len(groups)
        while i < n:
            current_group = groups[i]
            if len(current_group) < min_size:
                new_group = list(current_group)
                j = i + 1
                while j < n:
                    next_group = groups[j]
                    if len(next_group) < min_size and (len(new_group) + len(next_group)) <= max_size:
                        new_group.extend(next_group)
                        j += 1
                    else: 
                        break
                merged_groups.append(new_group)
                i = j
            else:
                merged_groups.append(current_group)
                i += 1
        return merged_groups


    def group_columns_by_rule(
        self,
        data_rows: List[List[Any]],
        formatted_header: List[str],
        sample_size: int = 10,
        lcs_threshold: int = 5,
        min_group_size: int = 3,
        max_group_size: int = 8,
    ) -> List[List[str]]:
        """
        Groups columns by data similarity and then merges small adjacent groups.

        Step 1: Performs a data-driven grouping based on Longest Common Substring
                similarity over a sample of the data.
        Step 2: Post-processes the result to merge adjacent groups that have fewer
                than 3 columns, ensuring the merged group does not exceed 8 columns.

        Args:
            data_rows: A list of data rows.
            formatted_header: A list of column names.
            sample_size: The number of rows to sample for comparison.
            lcs_threshold: The LCS length threshold for a value pair to be a match.
            match_count_threshold: The number of matching rows needed to link columns.

        Returns:
            A list of cleaned and pragmatically sized column groups.
        """
        if not formatted_header or not data_rows:
            return []

        # --- Step 1: Initial Grouping (same as before) ---
        actual_sample_size = min(len(data_rows), sample_size)
        if actual_sample_size == 0:
            return [[header] for header in formatted_header]
        
        sampled_rows: List[List[Any]] = random.sample(data_rows, actual_sample_size)
        num_cols: int = len(formatted_header)
        sampled_column_data: Dict[str, List[Any]] = {h: [] for h in formatted_header}
        for row in sampled_rows:
            if len(row) == num_cols:
                for i, header in enumerate(formatted_header):
                    sampled_column_data[header].append(row[i])

        initial_groups: List[List[str]] = []
        assigned_columns: Set[str] = set()

        for start_col_name in formatted_header:
            if start_col_name not in assigned_columns:
                current_group = []
                queue = collections.deque([start_col_name])
                assigned_columns.add(start_col_name)
                while queue:
                    current_col_name = queue.popleft()
                    current_group.append(current_col_name)
                    for candidate_col_name in formatted_header:
                        if candidate_col_name not in assigned_columns:
                            if self._are_columns_related(
                                sampled_column_data[current_col_name],
                                sampled_column_data[candidate_col_name],
                                lcs_threshold,
                            ):
                                assigned_columns.add(candidate_col_name)
                                queue.append(candidate_col_name)
                initial_groups.append(sorted(current_group))
        
        # --- Step 2: Merge the small groups ---
        groups_to_merge: List[List[str]] = [group for group in initial_groups if len(group) < min_group_size]
        final_groups: List[List[str]] = self._merge_groups_by_rule(groups_to_merge, min_size=min_group_size, max_size=max_group_size)
        for group in initial_groups:
            if len(group) >= min_group_size:
                final_groups.append(group)
        
        return final_groups


    def group_columns_by_semantic(
        self,
        data_rows: List[List[Any]],
        formatted_header: List[str],
        sheet_name: Optional[str] = None,
        sample_size: int = 5,
        max_retries: int = 3,
    ) -> List[List[str]]:
        """
        Group columns by their semantic meaning using LLM.
        
        Args:
            data_rows: List of data rows
            formatted_header: List of column names
            max_retries: Maximum number of retries
            
        Returns:
            List of groups, where each group is a list of column names
        """
        if len(formatted_header) <= 8:
            return [formatted_header]

        num_samples = min(sample_size, len(data_rows))
        sample_rows = random.sample(data_rows, num_samples)
        snippet = format_table_data_snippet_with_header(
            formatted_header=formatted_header,
            data_rows=sample_rows,
        )
        
        prompt = GROUP_COLUMNS_TEMPLATE.replace(
            "{{table_data_snippet}}", snippet
        ).replace(
            "{{sheet_name}}", sheet_name or "Unknown"
        )
        
        column_groups: List[List[str]] = []
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1,
                    extra_body = {
                        "chat_template_kwargs": {'enable_thinking': False},
                        "top_k": 20,
                        "mip_p": 0,
                    },
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty content")

                result = extract_json(content)
                
                # Validate the result
                if not result.get("column_groups"):
                    raise ValueError("Failed to group columns")
                
                column_groups = [group["columns"] for group in result["column_groups"]]
                
                # Validate that all columns are included and no duplicates
                all_columns_in_groups = []
                for group in column_groups:
                    all_columns_in_groups.extend(group)
                
                if set(all_columns_in_groups) != set(formatted_header):
                    raise ValueError("Column groups do not match the original header")
                
                if len(all_columns_in_groups) != len(set(all_columns_in_groups)):
                    raise ValueError("Duplicate columns found in groups")
                
                return column_groups
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                continue
        
        # Fallback: treat each column as its own group
        print("Failed to group columns, falling back to the last result")
        fixed_column_groups = []
        all_columns_in_groups = []
        for column_group in column_groups:
            fixed_column_group = []
            for column in column_group:
                if column in formatted_header:
                    fixed_column_group.append(column)
                    all_columns_in_groups.append(column)
            fixed_column_groups.append(fixed_column_group)
        for column in formatted_header:
            if column not in all_columns_in_groups:
                fixed_column_groups.append([column])
        return fixed_column_groups


    def group_columns_by_hybrid(
        self,
        data_rows: List[List[Any]],
        formatted_header: List[str],
        sheet_name: Optional[str] = None,
        sample_size: int = 20,
        lcs_threshold: int = 5,
        min_group_size: int = 3,
        max_group_size: int = 10,
        max_retries: int = 3,
    ) -> List[List[str]]:
        if len(formatted_header) <= max_group_size:
            return [formatted_header]
        
        groups_by_rule = self.group_columns_by_rule(
            data_rows=data_rows,
            formatted_header=formatted_header,
            sample_size=sample_size,
            lcs_threshold=lcs_threshold,
            min_group_size=1,
        )

        groups_by_llm = self.group_columns_by_semantic(
            data_rows=data_rows,
            formatted_header=formatted_header,
            sheet_name=sheet_name,
            sample_size=sample_size,
            max_retries=max_retries,
        )

        final_groups: List[List[str]] = []
        mergeable_rule_groups: List[List[str]] = []

        # --- Step 1: Segregate Rule Clusters ---
        for group in groups_by_rule:
            if len(group) > max_group_size:
                # If rule group is too big, keep it as is (per instructions)
                final_groups.append(group)
            else:
                mergeable_rule_groups.append(group)

        # --- Step 2: Map Columns to LLM Cluster IDs ---
        col_to_llm_id: Dict[str, int] = {}
        for idx, group in enumerate(groups_by_llm):
            for col in group:
                col_to_llm_id[col] = idx

        # --- Step 3: Assign Rule Clusters to LLM Buckets ---
        llm_buckets = collections.defaultdict(list)
        orphans = []

        for r_group in mergeable_rule_groups:
            # Vote for LLM group based on columns in this rule group
            votes = [col_to_llm_id[col] for col in r_group if col in col_to_llm_id]
            
            if not votes:
                orphans.append(r_group)
            else:
                # Majority vote wins
                most_common_llm_id = collections.Counter(votes).most_common(1)[0][0]
                llm_buckets[most_common_llm_id].append(r_group)

        # --- Step 4: Intra-Bucket Merging (Semantic Merging) ---
        # We create a list of "Candidates". Some might be valid size, some might be small.
        candidate_groups: List[List[str]] = []

        # Process buckets in order of LLM IDs to keep some stability
        sorted_llm_ids = sorted(llm_buckets.keys())
        
        # Combine orphans into the processing list as a "None" bucket or append at end
        # Let's process valid buckets first
        all_groups_to_process = [llm_buckets[i] for i in sorted_llm_ids]
        if orphans:
            all_groups_to_process.append(orphans)

        for rule_groups in all_groups_to_process:
            current_buffer: List[str] = []
            
            for group in rule_groups:
                # Can we add this group to buffer without exceeding MAX?
                if len(current_buffer) + len(group) <= max_group_size:
                    current_buffer.extend(group)
                else:
                    # Buffer is full/ready.
                    if current_buffer:
                        candidate_groups.append(current_buffer)
                    # Start new buffer with current group
                    current_buffer = list(group)
            
            # Flush the buffer for this bucket
            if current_buffer:
                candidate_groups.append(current_buffer)

        # --- Step 5: Enforce Min Group Size (The "Fixer" Pass) ---
        # We separate candidates into "Valid" (>= min) and "Undersized" (< min).
        
        valid_candidates = []
        undersized_candidates = []

        for group in candidate_groups:
            if len(group) >= min_group_size:
                valid_candidates.append(group)
            else:
                undersized_candidates.append(group)

        # Add valid ones to final immediately
        final_groups.extend(valid_candidates)

        # Now we must merge the undersized candidates together.
        # Since they are "leftovers" from different LLM buckets, they might 
        # not be semantically related, but we must satisfy min_size.
        
        leftover_buffer: List[str] = []
        
        for small_group in undersized_candidates:
            # Can we merge into leftover buffer?
            if len(leftover_buffer) + len(small_group) <= max_group_size:
                leftover_buffer.extend(small_group)
                
                # Check if we hit the "Sweet Spot" (>= min and <= max)
                # If we reached min_size, we *could* stop here to preserve some separation, but usually filling up slightly more is safer for packing. Here we finalize ONLY if we are fairly large or if we want to isolate. Let's simple-pack: keep adding until we HAVE to split or we finish.
            else:
                # Adding this would exceed max. But wait, is leftover_buffer valid?
                if len(leftover_buffer) >= min_group_size:
                    final_groups.append(leftover_buffer)
                    leftover_buffer = list(small_group)
                else:
                    # CRITICAL CASE: The leftover buffer is < min, AND we can't add the next group because it would make it > max. This implies `small_group` is huge  (which is impossible logically given it came from undersized list) OR max_size is extremely close to min_size. In this specific scenario, we force merge to avoid dropping data, or we finalize the undersized one (user preference: strictly >= min). Given inputs, undersized chunks are small. merging them usually fits.
                    final_groups.append(leftover_buffer)
                    leftover_buffer = list(small_group)
                    
        # Check the leftover buffer logic again: We want to finalize the buffer as soon as it hits min_group_size? Or keep it open? To fix "Cluster 6 and 7" problem, we should merge them. Revised Logic for Undersized: Just concactenate them all, then chunk by max_size (ensuring min_size).
        flat_undersized = [col for grp in undersized_candidates for col in grp]
        
        current_chunk = []
        for col in flat_undersized:
            current_chunk.append(col)
            # If we hit max size, dump it
            if len(current_chunk) == max_group_size:
                final_groups.append(current_chunk)
                current_chunk = []

        # --- Step 6: Handle the very last remnant ---
        if current_chunk:
            # If this chunk is >= min, great.
            if len(current_chunk) >= min_group_size:
                final_groups.append(current_chunk)
            else:
                # It is < min. We need to merge it into an existing group if possible. Try to merge into the last group in final_groups
                if final_groups:
                    last_group = final_groups[-1]
                    if len(last_group) + len(current_chunk) <= max_group_size:
                        # Merge into previous
                        last_group.extend(current_chunk)
                    else:
                        # Cannot merge into previous without violating max.
                        # We have no choice: either violate min (keep standalone) 
                        # or violate max (merge).
                        # Usually, violate max is worse for context windows. 
                        # We leave it as standalone (best effort).
                        final_groups.append(current_chunk)
                else:
                    # There were no other groups at all. Return what we have.
                    final_groups.append(current_chunk)

        return final_groups