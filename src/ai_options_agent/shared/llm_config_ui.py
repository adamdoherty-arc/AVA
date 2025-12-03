"""
Shared LLM Provider Configuration UI
Provides provider selection, testing, and management
"""

import streamlit as st
import os
from typing import Optional, List, Dict, Any


class LLMConfigUI:
    """Reusable LLM configuration interface"""

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    def render_provider_selector(
        self,
        show_add_provider: bool = True,
        allow_manual_selection: bool = True
    ) -> Optional[str]:
        """
        Render LLM provider selection UI

        Args:
            show_add_provider: Show "Add New Provider" expander
            allow_manual_selection: Allow user to select specific provider

        Returns:
            Selected provider ID or None (for auto-select)
        """
        st.subheader("🤖 LLM Provider Configuration")

        # Get available providers
        available_providers = self.llm_manager.get_available_providers()

        col1, col2 = st.columns([2, 1])

        selected_provider = None

        with col1:
            st.markdown("**Available Providers:**")

            if available_providers:
                if allow_manual_selection:
                    # Create provider selection
                    provider_options = {
                        f"{p['name']} - {p['cost']} (Speed: {p['speed']})": p['id']
                        for p in available_providers
                    }
                    provider_options["🔄 Auto-select (Prioritizes Free/Cheap)"] = None

                    selected_provider_display = st.selectbox(
                        "Choose LLM Provider:",
                        options=list(provider_options.keys()),
                        index=0,
                        help="Select which AI provider to use for reasoning"
                    )

                    selected_provider = provider_options[selected_provider_display]

                    # Show provider details
                    if selected_provider:
                        provider_info = next((p for p in available_providers if p['id'] == selected_provider), None)
                        if provider_info:
                            st.info(f"""
                            **{provider_info['name']}**
                            - Model: `{provider_info['current_model']}`
                            - Cost: {provider_info['cost']}
                            - Speed: {provider_info['speed']}
                            - Quality: {provider_info['quality']}
                            """)
                else:
                    # Just show available providers (no selection)
                    st.write(f"**{len(available_providers)} AI models ready:**")
                    for p in available_providers:
                        st.write(f"- ✅ **{p['name']}** - {p['description']} ({p['cost']})")
            else:
                st.warning("⚠️ No LLM providers available. Add API keys below to enable LLM reasoning.")

        with col2:
            st.markdown("**Provider Status:**")
            if available_providers:
                for p in available_providers[:3]:  # Show top 3
                    st.success(f"✓ {p['name']}")
            else:
                st.error("No providers configured")

        # Add new provider section
        if show_add_provider:
            self._render_add_provider_section()

        return selected_provider

    def render_simple_provider_list(self) -> None:
        """Simple provider list display (no selection)"""
        available_providers = self.llm_manager.get_available_providers()

        with st.expander("🤖 AI Models Active", expanded=False):
            st.write(f"**{len(available_providers)} AI models ready:**")
            for p in available_providers:
                st.write(f"- ✅ **{p['name']}** - {p['description']} ({p['cost']})")

    def _render_add_provider_section(self) -> None:
        """Render 'Add New Provider' expander"""
        with st.expander("➕ Add New LLM Provider", expanded=False):
            st.markdown("Add a new AI provider by entering its API key:")

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                new_provider_type = st.selectbox(
                    "Provider Type:",
                    ["OpenAI", "Anthropic Claude", "Google Gemini", "DeepSeek", "Groq", "Grok (xAI)", "Kimi/Moonshot"],
                    help="Select the type of provider you want to add"
                )

            with col2:
                new_api_key = st.text_input(
                    "API Key:",
                    type="password",
                    placeholder="sk-...",
                    help="Enter your API key for this provider"
                )

            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                test_provider = st.button("🧪 Test", use_container_width=True)

            if test_provider and new_api_key:
                self._test_provider(new_provider_type, new_api_key)

    def _test_provider(self, provider_type: str, api_key: str):
        """Test a new provider API key"""
        # Map display name to env var name
        provider_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic Claude": "ANTHROPIC_API_KEY",
            "Google Gemini": "GOOGLE_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "Groq": "GROQ_API_KEY",
            "Grok (xAI)": "GROK_API_KEY",
            "Kimi/Moonshot": "KIMI_API_KEY"
        }

        env_var = provider_map.get(provider_type)

        with st.spinner(f"Testing {provider_type}..."):
            # Temporarily set the API key
            original_key = os.getenv(env_var)
            os.environ[env_var] = api_key

            try:
                # Reload LLM manager with new key
                from src.ai_options_agent.llm_manager import LLMManager
                test_manager = LLMManager()

                # Try to generate with this provider
                provider_id = provider_type.lower().split()[0]
                result = test_manager.generate(
                    "Say 'test successful' if you can read this.",
                    provider_id=provider_id,
                    max_tokens=50
                )

                if result['text'] and len(result['text']) > 0:
                    st.success(f"✅ {provider_type} is working! API key is valid.")
                    st.info(f"Add this to your .env file:\n```\n{env_var}={api_key}\n```")

                    # Show test response
                    with st.expander("Test Response"):
                        st.write(result['text'])
                else:
                    st.error(f"❌ Test failed - No response from {provider_type}")

            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")

            finally:
                # Restore original key
                if original_key:
                    os.environ[env_var] = original_key
                else:
                    os.environ.pop(env_var, None)
