"""Deduplicate session IDs in each sample of LongMemEval with tracking of changes."""
import os
import json
import uuid


def reconstruct_deduplication_mapping(original_data, deduplicated_data):
    """
    Reconstruct the deduplication mapping by comparing original and deduplicated data.
    
    Args:
        original_data: The original data before deduplication
        deduplicated_data: The data after deduplication
    
    Returns:
        deduplication_mapping: List of dicts containing:
            - question_id: The entry's question ID
            - duplicated_indices: List of indices that were deduplicated
            - id_mapping: Dict mapping new_id -> original_id for deduplicated sessions
    """
    deduplication_mapping = []
    
    for orig_entry, dedup_entry in zip(original_data, deduplicated_data):
        # First, normalize the original IDs (apply the 'answer' -> 'noans' transformation)
        normalized_orig_ids = []
        for session_id, session_data in zip(orig_entry['haystack_session_ids'], orig_entry['haystack_sessions']):
            if 'answer' in session_id and all([not turn['has_answer'] for turn in [x for x in session_data if x['role'] == 'user']]):
                session_id = session_id.replace('answer', 'noans')
            normalized_orig_ids.append(session_id)
        
        dedup_ids = dedup_entry['haystack_session_ids']
        
        # Find differences between normalized original and deduplicated IDs
        duplicated_indices = []
        id_mapping = {}
        
        for idx, (norm_id, dedup_id) in enumerate(zip(normalized_orig_ids, dedup_ids)):
            if norm_id != dedup_id:
                # This ID was changed during deduplication
                duplicated_indices.append(idx)
                id_mapping[dedup_id] = norm_id
        
        # Only add to mapping if there were duplicates
        if duplicated_indices:
            deduplication_mapping.append({
                'question_id': orig_entry.get('question_id', 'unknown'),
                'duplicated_indices': duplicated_indices,
                'id_mapping': id_mapping
            })
    
    return deduplication_mapping



def deduplicate_with_tracking(data, deduplication_mapping=None):
    """
    Deduplicate session IDs and track the mapping. Reuses existing mappings when provided.
    
    Args:
        data: The data to deduplicate (will be modified in place)
        deduplication_mapping: Either:
            - None: Create new mappings from scratch
            - List of dicts (format from previous runs): Will be converted and reused
            - String (file path): Load the mapping from JSON file
    
    Returns:
        tuple: (updated_data, deduplication_mapping)
        - updated_data: The data with deduplicated session IDs
        - deduplication_mapping: List of dicts containing:
            - question_id: The entry's question ID
            - duplicated_indices: List of indices that were deduplicated
            - id_mapping: Dict mapping new_id -> original_id for deduplicated sessions
    """
    # Load mapping if it's a file path
    if isinstance(deduplication_mapping, str):
        if os.path.isfile(deduplication_mapping):
            with open(deduplication_mapping, 'r') as f:
                deduplication_mapping = json.load(f)
        else:
            deduplication_mapping = None
    
    # Convert existing mapping list to a dict of original_id -> new_id for efficient lookup
    # This allows us to reuse the same new_id for duplicate original_ids
    existing_mapping_dict = {}
    if deduplication_mapping:
        for entry in deduplication_mapping:
            # id_mapping is new_id -> original_id, we need to reverse it
            for new_id, original_id in entry['id_mapping'].items():
                if original_id not in existing_mapping_dict:
                    existing_mapping_dict[original_id] = []
                existing_mapping_dict[original_id].append(new_id)
    
    # Track new mappings to append to the list
    new_mapping_entries = []
    
    for entry in data:
        all_ids = []
        updated_ids = []
        duplicated_indices = []
        id_mapping = {}
        
        # Normalize IDs
        for idx, (session_id, session_data, session_date) in enumerate(zip(
            entry['haystack_session_ids'], 
            entry['haystack_sessions'], 
            entry['haystack_dates']
        )):
            if 'answer' in session_id and all([not turn['has_answer'] for turn in [x for x in session_data if x['role'] == 'user']]):
                session_id = session_id.replace('answer', 'noans')
            all_ids.append(session_id)

        # Check for duplicates
        if len(all_ids) != len(set(all_ids)):
            # Find duplicates
            seen = set()
            duplicates = set()
            for x in all_ids:
                if x in seen:
                    duplicates.add(x)
                else:
                    seen.add(x)
            
            for d in duplicates:
                assert 'answer' not in d, f"Duplicate ID '{d}' still contains 'answer'"
            
            # Track which new_ids we've already used for this entry to handle multiple duplicates
            used_new_ids_in_entry = {}
            seen_ids = set()
            
            for idx, session_id in enumerate(all_ids):
                if session_id in seen_ids:
                    # This is a duplicate
                    # Check if we have an existing mapping for this original_id
                    if session_id in existing_mapping_dict:
                        # Reuse existing mapping
                        # If we already used one mapping for this ID in this entry, use the next one
                        if session_id not in used_new_ids_in_entry:
                            used_new_ids_in_entry[session_id] = 0
                        else:
                            used_new_ids_in_entry[session_id] += 1
                        
                        mapping_index = used_new_ids_in_entry[session_id]
                        
                        if mapping_index < len(existing_mapping_dict[session_id]):
                            # Reuse existing new_id
                            new_id = existing_mapping_dict[session_id][mapping_index]
                        else:
                            # Need a new mapping - ran out of existing ones
                            new_id = f"{session_id}_{uuid.uuid4().hex[:4]}"
                            # Add to existing mapping for future reuse
                            existing_mapping_dict[session_id].append(new_id)
                    else:
                        # No existing mapping, create a new one
                        new_id = f"{session_id}_{uuid.uuid4().hex[:4]}"
                        # Add to existing mapping for future reuse
                        if session_id not in existing_mapping_dict:
                            existing_mapping_dict[session_id] = []
                        existing_mapping_dict[session_id].append(new_id)
                        used_new_ids_in_entry[session_id] = 0
                    
                    updated_ids.append(new_id)
                    duplicated_indices.append(idx)
                    id_mapping[new_id] = session_id
                else:
                    updated_ids.append(session_id)
                    seen_ids.add(session_id)
            
            # Update the entry with the new IDs
            entry['haystack_session_ids'] = updated_ids
            
            # Store the deduplication info for this entry
            new_mapping_entries.append({
                'question_id': entry.get('question_id', 'unknown'),
                'duplicated_indices': duplicated_indices,
                'id_mapping': id_mapping
            })
        else:
            # No duplicates, keep the original (potentially modified) IDs
            entry['haystack_session_ids'] = all_ids
    
    # Merge existing and new mappings
    if deduplication_mapping is None:
        final_mapping = new_mapping_entries
    else:
        # Append new entries to existing mapping
        final_mapping = deduplication_mapping + new_mapping_entries
    
    return data, final_mapping


if __name__ == '__main__':
    # Load data files
    mapping_file = 'data/longmemeval-cleaned/deduplication_mapping_updated.json'
    with open('data/longmemeval-cleaned/longmemeval_s_cleaned.json', 'r') as f:
        data_original = json.load(f)

    # Save deduplicated data and mapping
    data, new_mapping = deduplicate_with_tracking(data_original, mapping_file)
    
    with open('data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json', 'w') as f:
        json.dump(data, f, indent=2)

    with open(mapping_file, 'w') as f:
        json.dump(new_mapping, f, indent=2)

    with open('data/longmemeval-cleaned/longmemeval_oracle.json', 'r') as f:
        data_original = json.load(f)
    data, new_mapping = deduplicate_with_tracking(data_original, mapping_file)
    with open('data/longmemeval-cleaned/longmemeval_oracle_deduplicate.json', 'w') as f:
        json.dump(data, f, indent=2)

    with open('data/longmemeval-cleaned/longmemeval_s_cleaned_deduplicate.json', 'r') as f:
        data_deduplicate = json.load(f)

    # # ============================================================
    # # RECONSTRUCT MAPPING FROM EXISTING DEDUPLICATED DATA
    # # ============================================================
    # print("="*60)
    # print("Reconstructing deduplication mapping from existing data...")
    # print("="*60)

    # # Compare original and deduplicated data to reconstruct the mapping
    # reconstructed_mapping = reconstruct_deduplication_mapping(data_original, data_deduplicate)

    # # Save the reconstructed mapping
    # reconstructed_mapping_file = 'data/longmemeval-cleaned/deduplication_mapping.json'
    # with open(reconstructed_mapping_file, 'w') as f:
    #     json.dump(reconstructed_mapping, f, indent=2)

    # print(f"\n✓ Reconstructed mapping saved to: {reconstructed_mapping_file}")
    # print(f"✓ Found {len(reconstructed_mapping)} entries with duplicates")

    # # Print summary
    # if reconstructed_mapping:
    #     total_duplicates = sum(len(entry['duplicated_indices']) for entry in reconstructed_mapping)
    #     print(f"✓ Total duplicated sessions renamed: {total_duplicates}")
    #     print("\n" + "="*60)
    #     print("Sample mappings (first 3 entries):")
    #     print("="*60)
    #     for entry in reconstructed_mapping[:3]:
    #         print(f"\nQuestion ID: {entry['question_id']}")
    #         print(f"  Duplicated indices: {entry['duplicated_indices']}")
    #         print(f"  ID mappings:")
    #         for new_id, orig_id in entry['id_mapping'].items():
    #             print(f"    {new_id} ← {orig_id}")
    # else:
    #     print("✓ No duplicates found in the data")
