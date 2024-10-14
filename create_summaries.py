import os
import git

import argparse
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# Module Constants
load_dotenv()
DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
BASE_URL = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints"

def generate_summary(commit, diff_text, base_url=BASE_URL, databricks_token=DATABRICKS_TOKEN):
    """
    Generate a summary for a given commit using OpenAI's API.
    
    Parameters:
        commit (git.Commit): The commit object to summarize.
        diff_text (str): The diff text for the commit.
        base_url (str): The base URL for the OpenAI API.
        databricks_token (str): The API token for accessing Databricks.
    
    Returns:
        str: Summary of the commit.
    """
    client = OpenAI(api_key=databricks_token, base_url=base_url)

    messages = [
        {
            "role": "system",
            "content": (
                "You are the bot which can parse Git commit's history and give a summary "
                "about the functionality added as part of the commit. You will be given the commit message, "
                "commit diff, and commit ID. In the output, give the author, summary, and commit ID and pull request Hashcode if possible. "
                "Make sure the summary is not more than 2-3 lines. Start the completion with Date."
            )
        },
        {
            "role": "user",
            "content": (
                f"--- Date ---\n{commit.committed_datetime}\n"
                f"--- Message ---\n{commit.message}\n"
                f"--- Commit ID ---\n{commit.hexsha}\n"
                f"--- Commit Diff ---\n{diff_text}"
            )
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="databricks-meta-llama-3-1-70b-instruct",
        max_tokens=512
    )

    return chat_completion.choices[0].message.content


def collect_commit_summaries(repo_path, after_date, user):
    """
    Collect commit summaries for a given repository.
    
    Parameters:
        repo_path (str): The path to the Git repository.
        after_date (str): The date after which commits should be considered.
        user (str): The author of the commits to filter.
    
    Returns:
        list: A list of dictionaries containing commit information and generated summaries.
    """
    repo = git.Repo(repo_path)
    commits_with_diff = []

    for commit in repo.iter_commits(since=after_date):
        if commit.author.name == user:
            # Get diffs for the commit
            diffs = []
            if commit.parents:
                # Compare with first parent
                diffs = commit.diff(commit.parents[0], create_patch=True)
            else:
                # Initial commit (no parents)
                diffs = commit.diff(git.Object.NULL_TREE, create_patch=True)

            # Collect diff text
            diff_text = ''
            for diff in diffs:
                try:
                    diff_text += diff.diff.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    diff_text += "<Binary data>\n"

            commits_with_diff.append({
                'date': commit.committed_datetime.strftime(DATE_TIME_FORMAT),
                'message': commit.message,
                'commit_hash': commit.hexsha,
                'diff': diff_text,
                'generated_summary': generate_summary(commit, diff_text)
            })

    return commits_with_diff


def main():
    parser = argparse.ArgumentParser(description="Generate summaries for Git commits.")
    parser.add_argument("--repo_path", type=str, default = '/Users/puneet.jain/Documents/GitHub/ray',
                         help="Path to the Git repository.")
    parser.add_argument("--after_date", type=str, default = '2024-02-15',
                        help="Date after which commits should be considered (YYYY-MM-DD format).")
    parser.add_argument("--user", type=str,  default = 'WeichenXu',
                        help="Author of the commits to filter.")
    args = parser.parse_args()

    # Collect commit summaries
    commits_with_diff = collect_commit_summaries(args.repo_path, args.after_date, args.user)

    # Write only the generated summaries to a text file
    output_file = os.path.join(os.getcwd(), "commit_summaries.txt")
    with open(output_file, "w") as f:
        for commit in commits_with_diff:
            f.write(commit['generated_summary'] + "\n")

    print(f"Generated summaries written to {output_file}")


if __name__ == "__main__":
    main()